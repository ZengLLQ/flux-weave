"""Stoichiometric matrix kernel analysis, conserved moieties, and thermodynamic cycle detection.

Computes:
1. Left null space ker(S^T): Conserved chemical moieties (e.g. ATP+ADP+AMP, NAD+NADH pools).
2. Right null space ker(S): Fundamental pathways spanning steady-state flux space.
3. Futile thermodynamic cycles violating the Second Law of Thermodynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from flux_weave.model import MetabolicModel


@dataclass
class ConservedMoiety:
    """Conserved biochemical moiety vector l such that l^T * S = 0."""

    id: str
    metabolite_coefficients: Dict[str, float]

    def formula_str(self) -> str:
        """Format as linear combination of metabolites."""
        terms = [
            f"{coeff:+g} * {m_id}"
            for m_id, coeff in sorted(self.metabolite_coefficients.items())
            if abs(coeff) > 1e-5
        ]
        return " ".join(terms) if terms else "0"


@dataclass
class KernelReport:
    """Full algebraic decomposition of stoichiometric matrix S."""

    rank: int
    num_metabolites: int
    num_reactions: int
    degrees_of_freedom: int
    conserved_moieties: List[ConservedMoiety]
    right_null_dimension: int


def compute_conserved_moieties(
    model: MetabolicModel,
    internal_only: bool = True,
    singular_tolerance: float = 1e-10,
) -> List[ConservedMoiety]:
    """Calculate basis of conserved moieties (left null space of stoichiometric matrix).

    Parameters:
        model: Target MetabolicModel.
        internal_only: If True, evaluates S without exchange reactions to detect closed pools.
        singular_tolerance: SVD singular value cutoff.

    Returns:
        List of ConservedMoiety instances representing invariant conservation pools.
    """
    if internal_only:
        # Filter out exchange reactions
        rxns = [r for r in model.reactions if not r.is_exchange]
    else:
        rxns = model.reactions

    mets = model.metabolites
    if not rxns or not mets:
        return []

    # Build sub-matrix S_int
    m = len(mets)
    n = len(rxns)
    s_mat = np.zeros((m, n), dtype=np.float64)
    met_map = {m_obj.id: i for i, m_obj in enumerate(mets)}

    for j, rxn in enumerate(rxns):
        for met_obj, coeff in rxn.metabolites.items():
            if met_obj.id in met_map:
                s_mat[met_map[met_obj.id], j] = coeff

    # Left null space: ker(S^T) --> S^T * l = 0
    # SVD of S^T (n x m)
    _, s_vals, v_h = np.linalg.svd(s_mat.T, full_matrices=True)
    null_mask = np.ones(m, dtype=bool)
    if len(s_vals) > 0:
        null_mask[: len(s_vals)] = s_vals <= singular_tolerance

    left_null_vectors = v_h[null_mask, :]

    moieties: List[ConservedMoiety] = []
    for k, vec in enumerate(left_null_vectors):
        # Normalize and filter near-zeros
        max_abs = float(np.max(np.abs(vec)))
        if max_abs < 1e-7:
            continue
        norm_vec = vec / max_abs

        # Attempt to scale to integer coefficients if close to rational
        min_nonzero = float(np.min(np.abs(norm_vec[np.abs(norm_vec) > 1e-4])))
        if min_nonzero > 1e-5:
            scaled = norm_vec / min_nonzero
            if np.allclose(scaled, np.round(scaled), atol=1e-3):
                norm_vec = np.round(scaled)

        coeff_dict = {
            mets[i].id: float(norm_vec[i])
            for i in range(m)
            if abs(norm_vec[i]) > 1e-5
        }
        if coeff_dict:
            moieties.append(
                ConservedMoiety(id=f"MOIETY_{k+1}", metabolite_coefficients=coeff_dict)
            )

    return moieties


def analyze_stoichiometric_kernel(model: MetabolicModel) -> KernelReport:
    """Perform complete structural rank and nullspace decomposition on model."""
    s_mat = model.stoichiometric_matrix()
    m, n = s_mat.shape

    u, s_vals, v_h = np.linalg.svd(s_mat, full_matrices=True)
    rank = int(np.sum(s_vals > 1e-10))
    dof = n - rank

    moieties = compute_conserved_moieties(model, internal_only=True)

    return KernelReport(
        rank=rank,
        num_metabolites=m,
        num_reactions=n,
        degrees_of_freedom=dof,
        conserved_moieties=moieties,
        right_null_dimension=dof,
    )


def detect_futile_cycles(
    model: MetabolicModel,
    tolerance: float = 1e-7,
) -> List[Tuple[str, float]]:
    """Detect presence of unconstrained internal stoichiometric loops carrying flux."""
    from scipy.optimize import linprog  # type: ignore

    # Zero all exchange reaction capacities
    model_copy = model.copy()
    for rxn in model_copy.reactions:
        if rxn.is_exchange:
            rxn.lower_bound = 0.0
            rxn.upper_bound = 0.0

    # Maximize sum of internal fluxes
    rxns = model_copy.reactions
    c_lp = -np.ones(len(rxns), dtype=np.float64)
    s_mat = model_copy.stoichiometric_matrix()
    b_eq = np.zeros(s_mat.shape[0], dtype=np.float64)
    lb, ub = model_copy.get_bounds()
    bounds = [(float(lower), float(upper)) for lower, upper in zip(lb, ub)]

    res = linprog(
        c=c_lp,
        A_eq=s_mat,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return []

    active_loops = [
        (rxns[i].id, float(res.x[i]))
        for i in range(len(rxns))
        if abs(res.x[i]) > tolerance
    ]
    return active_loops
