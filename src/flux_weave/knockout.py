"""In silico Gene and Reaction Deletion screening engine.

Simulates genetic perturbations, identifies essential genes/reactions,
and screens for Synthetic Lethality in metabolic networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from flux_weave.fba import solve_fba
from flux_weave.gpr import eval_gpr
from flux_weave.model import MetabolicModel


class EssentialityCategory(str, Enum):
    """Classification of gene or reaction essentiality."""

    ESSENTIAL = "essential"
    SUBLETHAL = "sublethal"
    NON_ESSENTIAL = "non_essential"


@dataclass
class DeletionResult:
    """Outcome of an in silico knockout experiment."""

    entity_ids: Tuple[str, ...]
    wild_type_growth: float
    knockout_growth: float
    growth_fraction: float
    category: EssentialityCategory
    disabled_reactions: List[str]

    @property
    def is_synthetic_lethal(self) -> bool:
        """True if pairwise deletion causes lethality when individual deletions do not."""
        return len(self.entity_ids) == 2 and self.category == EssentialityCategory.ESSENTIAL


def classify_growth(growth_fraction: float, lethal_cutoff: float = 0.01) -> EssentialityCategory:
    """Classify relative growth rate into essentiality category."""
    if growth_fraction < lethal_cutoff:
        return EssentialityCategory.ESSENTIAL
    elif growth_fraction < 0.95:
        return EssentialityCategory.SUBLETHAL
    return EssentialityCategory.NON_ESSENTIAL


def single_reaction_deletion(
    model: MetabolicModel,
    reaction_ids: Optional[List[str]] = None,
    tolerance: float = 1e-8,
) -> Dict[str, DeletionResult]:
    """Screen single reaction knockouts across the metabolic model."""
    wt_sol = solve_fba(model, tolerance=tolerance)
    if not wt_sol.is_optimal:
        raise RuntimeError("Wild-type model is not viable. Cannot perform deletion study.")

    wt_growth = wt_sol.objective_value
    targets = reaction_ids if reaction_ids is not None else [r.id for r in model.reactions]

    results: Dict[str, DeletionResult] = {}
    model_copy = model.copy()

    for r_id in targets:
        rxn = model_copy.get_reaction(r_id)
        orig_lb, orig_ub = rxn.lower_bound, rxn.upper_bound

        # Knockout reaction: zero capacity
        rxn.lower_bound = 0.0
        rxn.upper_bound = 0.0

        sol = solve_fba(model_copy, tolerance=tolerance)
        ko_growth = sol.objective_value if sol.is_optimal else 0.0
        fraction = max(0.0, ko_growth / wt_growth) if wt_growth > 1e-9 else 0.0
        category = classify_growth(fraction)

        results[r_id] = DeletionResult(
            entity_ids=(r_id,),
            wild_type_growth=wt_growth,
            knockout_growth=ko_growth,
            growth_fraction=fraction,
            category=category,
            disabled_reactions=[r_id],
        )

        # Restore reaction bounds
        rxn.lower_bound = orig_lb
        rxn.upper_bound = orig_ub

    return results


def single_gene_deletion(
    model: MetabolicModel,
    gene_ids: Optional[List[str]] = None,
    tolerance: float = 1e-8,
) -> Dict[str, DeletionResult]:
    """Screen single gene knockouts using GPR Boolean association rules."""
    wt_sol = solve_fba(model, tolerance=tolerance)
    if not wt_sol.is_optimal:
        raise RuntimeError("Wild-type model is not viable. Cannot perform gene knockout study.")

    wt_growth = wt_sol.objective_value
    all_genes = sorted(list(model.extract_all_genes()))
    targets = gene_ids if gene_ids is not None else all_genes

    results: Dict[str, DeletionResult] = {}
    model_copy = model.copy()

    for g_id in targets:
        knocked_set: Set[str] = {g_id}
        disabled_rxns: List[str] = []
        orig_bounds: Dict[str, Tuple[float, float]] = {}

        for rxn in model_copy.reactions:
            if rxn.gene_reaction_rule:
                is_active = eval_gpr(rxn.gene_reaction_rule, knocked_set, is_knockout_set=True)
                if not is_active:
                    disabled_rxns.append(rxn.id)
                    orig_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
                    rxn.lower_bound = 0.0
                    rxn.upper_bound = 0.0

        sol = solve_fba(model_copy, tolerance=tolerance)
        ko_growth = sol.objective_value if sol.is_optimal else 0.0
        fraction = max(0.0, ko_growth / wt_growth) if wt_growth > 1e-9 else 0.0
        category = classify_growth(fraction)

        results[g_id] = DeletionResult(
            entity_ids=(g_id,),
            wild_type_growth=wt_growth,
            knockout_growth=ko_growth,
            growth_fraction=fraction,
            category=category,
            disabled_reactions=disabled_rxns,
        )

        # Restore bounds
        for rxn_id, (old_lb, old_ub) in orig_bounds.items():
            r = model_copy.get_reaction(rxn_id)
            r.lower_bound = old_lb
            r.upper_bound = old_ub

    return results


def double_gene_deletion(
    model: MetabolicModel,
    gene_pairs: Optional[List[Tuple[str, str]]] = None,
    tolerance: float = 1e-8,
) -> Dict[Tuple[str, str], DeletionResult]:
    """Screen double gene knockouts to detect Synthetic Lethality."""
    wt_sol = solve_fba(model, tolerance=tolerance)
    if not wt_sol.is_optimal:
        raise RuntimeError("Wild-type model is not viable.")

    wt_growth = wt_sol.objective_value
    model_copy = model.copy()

    # If no pairs provided, pair non-essential genes from single deletion
    if gene_pairs is None:
        single_res = single_gene_deletion(model, tolerance=tolerance)
        non_essentials = [
            g for g, res in single_res.items()
            if res.category == EssentialityCategory.NON_ESSENTIAL
        ]
        gene_pairs = []
        for i in range(len(non_essentials)):
            for j in range(i + 1, len(non_essentials)):
                gene_pairs.append((non_essentials[i], non_essentials[j]))

    results: Dict[Tuple[str, str], DeletionResult] = {}

    for g1, g2 in gene_pairs:
        knocked_set: Set[str] = {g1, g2}
        disabled_rxns: List[str] = []
        orig_bounds: Dict[str, Tuple[float, float]] = {}

        for rxn in model_copy.reactions:
            if rxn.gene_reaction_rule:
                is_active = eval_gpr(rxn.gene_reaction_rule, knocked_set, is_knockout_set=True)
                if not is_active:
                    disabled_rxns.append(rxn.id)
                    orig_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
                    rxn.lower_bound = 0.0
                    rxn.upper_bound = 0.0

        sol = solve_fba(model_copy, tolerance=tolerance)
        ko_growth = sol.objective_value if sol.is_optimal else 0.0
        fraction = max(0.0, ko_growth / wt_growth) if wt_growth > 1e-9 else 0.0
        category = classify_growth(fraction)

        pair_key = (g1, g2)
        results[pair_key] = DeletionResult(
            entity_ids=pair_key,
            wild_type_growth=wt_growth,
            knockout_growth=ko_growth,
            growth_fraction=fraction,
            category=category,
            disabled_reactions=disabled_rxns,
        )

        for rxn_id, (old_lb, old_ub) in orig_bounds.items():
            r = model_copy.get_reaction(rxn_id)
            r.lower_bound = old_lb
            r.upper_bound = old_ub

    return results
