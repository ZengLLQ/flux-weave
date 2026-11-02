"""Flux Variability Analysis (FVA) engine.

Computes the minimum and maximum possible flux through every metabolic reaction
while enforcing that the biological objective meets or exceeds a target fraction
of the theoretical maximum:
    c^T v >= gamma * Z*

Identifies essential reactions, metabolic redundancies, and blocked pathways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import linprog  # type: ignore

from flux_weave.fba import solve_fba
from flux_weave.model import MetabolicModel


@dataclass
class FVAResult:
    """Individual reaction variability interval."""

    reaction_id: str
    minimum: float
    maximum: float

    @property
    def span(self) -> float:
        """Flux plasticity interval width: maximum - minimum."""
        return self.maximum - self.minimum

    @property
    def is_blocked(self) -> bool:
        """Return True if reaction carries zero flux across all optimal solutions."""
        return abs(self.minimum) < 1e-6 and abs(self.maximum) < 1e-6

    @property
    def is_essential(self) -> bool:
        """Return True if reaction cannot carry zero flux without reducing growth."""
        return self.minimum > 1e-6 or self.maximum < -1e-6


@dataclass
class FVASolution:
    """Complete Flux Variability Analysis dataset across all requested reactions."""

    results: Dict[str, FVAResult]
    fraction_optimum: float
    objective_value: float

    def get_span(self, reaction_id: str) -> float:
        """Retrieve variability span for a specific reaction."""
        return self.results[reaction_id].span if reaction_id in self.results else 0.0

    @property
    def essential_reactions(self) -> List[str]:
        """List reactions that must carry non-zero flux to maintain target objective."""
        return [r_id for r_id, res in self.results.items() if res.is_essential]

    @property
    def blocked_reactions(self) -> List[str]:
        """List reactions unable to carry any flux under optimal conditions."""
        return [r_id for r_id, res in self.results.items() if res.is_blocked]

    @property
    def plastic_reactions(self) -> List[str]:
        """List reactions possessing non-zero flux variability (span > 1e-6)."""
        return [r_id for r_id, res in self.results.items() if res.span > 1e-6]


def run_fva(
    model: MetabolicModel,
    reaction_ids: Optional[List[str]] = None,
    fraction_optimum: float = 1.0,
    tolerance: float = 1e-8,
) -> FVASolution:
    """Execute Flux Variability Analysis on specified reactions.

    Parameters:
        model: Target MetabolicModel.
        reaction_ids: Optional subset of reaction IDs to analyze. If None, analyzes all reactions.
        fraction_optimum: Required fraction gamma in [0.0, 1.0] of maximum objective value.
        tolerance: Solver numerical tolerance.

    Returns:
        FVASolution containing min, max, and span for each reaction.
    """
    if fraction_optimum < 0.0 or fraction_optimum > 1.0:
        raise ValueError(
            f"fraction_optimum must be between 0.0 and 1.0, got {fraction_optimum}"
        )

    # Step 1: Initial FBA to establish benchmark objective Z*
    baseline_fba = solve_fba(model, tolerance=tolerance)
    if not baseline_fba.is_optimal:
        raise RuntimeError(
            f"Baseline FBA failed with status '{baseline_fba.status}'. Cannot perform FVA."
        )

    z_opt = baseline_fba.objective_value
    target_cutoff = fraction_optimum * z_opt

    rxns = model.reactions
    rxn_ids_all = [r.id for r in rxns]
    targets = reaction_ids if reaction_ids is not None else rxn_ids_all

    s_mat = model.stoichiometric_matrix()
    b_eq = np.zeros(s_mat.shape[0], dtype=np.float64)
    lb, ub = model.get_bounds()
    bounds = [(float(lower), float(upper)) for lower, upper in zip(lb, ub)]

    # Inequality constraint: c^T v >= gamma * Z*  -->  -c^T v <= -gamma * Z*
    c_obj = model.get_objective_vector()
    a_ub = -c_obj.reshape(1, -1)
    b_ub = np.array([-target_cutoff], dtype=np.float64)

    options = {
        "primal_feasibility_tolerance": tolerance,
        "dual_feasibility_tolerance": tolerance,
    }

    results: Dict[str, FVAResult] = {}
    rxn_index_map = {r.id: i for i, r in enumerate(rxns)}

    for r_id in targets:
        if r_id not in rxn_index_map:
            continue
        idx = rxn_index_map[r_id]

        # Minimize v_idx: c_lp has +1 at idx
        c_min = np.zeros(len(rxns), dtype=np.float64)
        c_min[idx] = 1.0
        res_min = linprog(
            c=c_min,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=s_mat,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )
        v_min = float(res_min.x[idx]) if res_min.success else float(lb[idx])

        # Maximize v_idx: c_lp has -1 at idx
        c_max = np.zeros(len(rxns), dtype=np.float64)
        c_max[idx] = -1.0
        res_max = linprog(
            c=c_max,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=s_mat,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )
        v_max = float(res_max.x[idx]) if res_max.success else float(ub[idx])

        # Numerical sanity check
        if v_min > v_max:
            # If numerical noise made v_min slightly > v_max, collapse to average
            if abs(v_min - v_max) < 1e-5:
                v_avg = 0.5 * (v_min + v_max)
                v_min, v_max = v_avg, v_avg
            else:
                v_min, v_max = v_max, v_min

        results[r_id] = FVAResult(reaction_id=r_id, minimum=v_min, maximum=v_max)

    return FVASolution(
        results=results,
        fraction_optimum=fraction_optimum,
        objective_value=z_opt,
    )
