"""Parsimonious Flux Balance Analysis (pFBA) engine.

Eliminates thermodynamically infeasible futile cycles and identifies the minimal
intracellular enzyme burden (Lewis et al., 2010):
    minimize    sum_{j=1}^n |v_j|
    subject to  S * v = 0
                c^T v >= gamma * Z*
                v_min <= v <= v_max

Formulated as an augmented Linear Program with auxiliary slack variables u_j >= |v_j|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog  # type: ignore

from flux_weave.fba import solve_fba
from flux_weave.model import MetabolicModel


@dataclass
class PFBASolution:
    """Parsimonious FBA solution containing optimal flux and total enzyme burden."""

    status: str
    objective_value: float
    total_flux_burden: float
    fluxes: Dict[str, float]

    @property
    def is_optimal(self) -> bool:
        """Return True if pFBA solved to optimality."""
        return self.status == "optimal"

    def get_flux(self, reaction_id: str) -> float:
        """Retrieve flux rate for a specific reaction."""
        return self.fluxes.get(reaction_id, 0.0)

    def active_reactions(self, threshold: float = 1e-6) -> List[Tuple[str, float]]:
        """Return list of reactions carrying significant flux."""
        return [(r, f) for r, f in self.fluxes.items() if abs(f) > threshold]


def solve_pfba(
    model: MetabolicModel,
    fraction_optimum: float = 1.0,
    tolerance: float = 1e-8,
) -> PFBASolution:
    """Solve Parsimonious Flux Balance Analysis on the model.

    Parameters:
        model: Target MetabolicModel.
        fraction_optimum: Fraction of baseline optimal growth rate to retain (default 1.0).
        tolerance: Solver numerical tolerance.

    Returns:
        PFBASolution with minimal total enzyme burden and cycle-free flux distribution.
    """
    # Step 1: Baseline FBA
    baseline = solve_fba(model, tolerance=tolerance)
    if not baseline.is_optimal:
        return PFBASolution(
            status=f"baseline_failed_{baseline.status}",
            objective_value=0.0,
            total_flux_burden=0.0,
            fluxes={},
        )

    z_opt = baseline.objective_value
    target_cutoff = fraction_optimum * z_opt

    rxns = model.reactions
    mets = model.metabolites
    n = len(rxns)
    m = len(mets)

    # Augmented variables x = [v (n); u (n)] in R^(2n)
    # Objective: min sum(u_j) --> c_pfba = [0_n, 1_n]
    c_pfba = np.zeros(2 * n, dtype=np.float64)
    c_pfba[n:] = 1.0

    # Equality constraints: S * v = 0 --> [S, 0] * [v; u] = 0
    s_mat = model.stoichiometric_matrix()
    a_eq = np.zeros((m, 2 * n), dtype=np.float64)
    a_eq[:, :n] = s_mat
    b_eq = np.zeros(m, dtype=np.float64)

    # Inequality constraints:
    # 1) v_j - u_j <= 0  -->  [I_n, -I_n] * [v; u] <= 0
    # 2) -v_j - u_j <= 0 -->  [-I_n, -I_n] * [v; u] <= 0
    # 3) -c^T v <= -target_cutoff --> [-c_obj, 0_n] * [v; u] <= -target_cutoff
    a_ub = np.zeros((2 * n + 1, 2 * n), dtype=np.float64)
    b_ub = np.zeros(2 * n + 1, dtype=np.float64)

    # v_j - u_j <= 0
    a_ub[:n, :n] = np.eye(n)
    a_ub[:n, n:] = -np.eye(n)

    # -v_j - u_j <= 0
    a_ub[n : 2 * n, :n] = -np.eye(n)
    a_ub[n : 2 * n, n:] = -np.eye(n)

    # Growth cutoff
    c_obj = model.get_objective_vector()
    a_ub[2 * n, :n] = -c_obj
    b_ub[2 * n] = -target_cutoff

    # Variable bounds
    lb, ub = model.get_bounds()
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    # Bounds for v
    for lower, upper in zip(lb, ub):
        bounds.append((float(lower), float(upper)))
    # Bounds for u (u_j >= 0)
    for lower, upper in zip(lb, ub):
        max_bound = max(abs(float(lower)), abs(float(upper)))
        bounds.append((0.0, max_bound))

    options = {
        "primal_feasibility_tolerance": tolerance,
        "dual_feasibility_tolerance": tolerance,
    }

    res = linprog(
        c=c_pfba,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options=options,
    )

    if not res.success:
        return PFBASolution(
            status=f"pfba_failed_{res.status}",
            objective_value=z_opt,
            total_flux_burden=0.0,
            fluxes=baseline.fluxes,
        )

    v_sol = res.x[:n]
    u_sol = res.x[n:]
    total_burden = float(np.sum(u_sol))

    flux_dict = {rxns[i].id: float(v_sol[i]) for i in range(n)}

    return PFBASolution(
        status="optimal",
        objective_value=z_opt,
        total_flux_burden=total_burden,
        fluxes=flux_dict,
    )
