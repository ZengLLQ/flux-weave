"""Flux Balance Analysis (FBA) linear programming solver.

Formulation:
    maximize    Z = c^T v
    subject to  S * v = 0               (steady-state mass conservation)
                v_min <= v <= v_max     (thermodynamic and capacity bounds)

Solves via the HiGHS dual simplex/interior-point linear programming engine.
Extracts optimal flux vectors, metabolite shadow prices (dual values),
and reaction reduced costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.optimize import linprog  # type: ignore

from flux_weave.model import MetabolicModel


@dataclass
class FBASolution:
    """Complete solution record from a Flux Balance Analysis run."""

    status: str
    objective_value: float
    fluxes: Dict[str, float]
    shadow_prices: Dict[str, float]
    reduced_costs: Dict[str, float]

    @property
    def is_optimal(self) -> bool:
        """Return True if the LP reached optimal termination."""
        return self.status == "optimal"

    def get_flux(self, reaction_id: str) -> float:
        """Retrieve flux rate for a specific reaction."""
        return self.fluxes.get(reaction_id, 0.0)

    def get_shadow_price(self, metabolite_id: str) -> float:
        """Retrieve shadow price (dual marginal value) for a specific metabolite."""
        return self.shadow_prices.get(metabolite_id, 0.0)

    def top_fluxes(self, n: int = 10, abs_sort: bool = True) -> List[tuple[str, float]]:
        """Return top n reactions ranked by flux magnitude."""
        items = list(self.fluxes.items())
        if abs_sort:
            items.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]


def solve_fba(
    model: MetabolicModel,
    method: str = "highs",
    tolerance: float = 1e-9,
) -> FBASolution:
    """Solve Flux Balance Analysis LP for the given metabolic model.

    Parameters:
        model: Target MetabolicModel with populated stoichiometry, bounds, and objective.
        method: LP solver method ('highs', 'highs-ds', or 'highs-ipm'). Default is 'highs'.
        tolerance: Feasibility and optimality tolerance.

    Returns:
        FBASolution containing status, objective value, fluxes, shadow prices, and reduced costs.
    """
    rxns = model.reactions
    mets = model.metabolites

    if not rxns or not mets:
        return FBASolution(
            status="empty_model",
            objective_value=0.0,
            fluxes={},
            shadow_prices={},
            reduced_costs={},
        )

    # Formulation: linprog minimizes c_lp^T v, so c_lp = -c_obj
    c_obj = model.get_objective_vector()
    c_lp = -c_obj

    # Equality constraints: S * v = 0
    s_mat = model.stoichiometric_matrix()
    b_eq = np.zeros(s_mat.shape[0], dtype=np.float64)

    # Reaction bounds
    lb, ub = model.get_bounds()
    bounds = [(float(lower), float(upper)) for lower, upper in zip(lb, ub)]

    # Optimize LP
    options = {"primal_feasibility_tolerance": tolerance, "dual_feasibility_tolerance": tolerance}
    res = linprog(
        c=c_lp,
        A_eq=s_mat,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        options=options,
    )

    if not res.success:
        status_msg = (
            "infeasible"
            if res.status == 2
            else "unbounded"
            if res.status == 3
            else f"failed_{res.status}"
        )
        return FBASolution(
            status=status_msg,
            objective_value=0.0,
            fluxes={r.id: 0.0 for r in rxns},
            shadow_prices={m.id: 0.0 for m in mets},
            reduced_costs={r.id: 0.0 for r in rxns},
        )

    opt_obj = -float(res.fun)
    flux_vector = res.x
    flux_dict = {rxns[i].id: float(flux_vector[i]) for i in range(len(rxns))}

    # Extract dual values (shadow prices and reduced costs)
    shadow_dict: Dict[str, float] = {}
    reduced_dict: Dict[str, float] = {}

    marginals = getattr(res, "marginals", None)
    if marginals is not None:
        eq_marginals = getattr(marginals, "equality", None)
        if eq_marginals is not None and len(eq_marginals) == len(mets):
            # Since linprog minimized -c^T v, flip sign for true shadow price dZ/db
            shadow_dict = {mets[i].id: -float(eq_marginals[i]) for i in range(len(mets))}
        else:
            shadow_dict = {m.id: 0.0 for m in mets}

        lower_marg = getattr(marginals, "lower", None)
        upper_marg = getattr(marginals, "upper", None)
        if lower_marg is not None and upper_marg is not None:
            reduced_dict = {
                rxns[i].id: float(lower_marg[i]) - float(upper_marg[i])
                for i in range(len(rxns))
            }
        else:
            reduced_dict = {r.id: 0.0 for r in rxns}
    else:
        shadow_dict = {m.id: 0.0 for m in mets}
        reduced_dict = {r.id: 0.0 for r in rxns}

    return FBASolution(
        status="optimal",
        objective_value=opt_obj,
        fluxes=flux_dict,
        shadow_prices=shadow_dict,
        reduced_costs=reduced_dict,
    )
