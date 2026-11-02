"""Unit tests for stoichiometric matrix null space and conserved moieties."""

import numpy as np
from flux_weave.canonical import create_toy_linear_model
from flux_weave.model import MetabolicModel, Metabolite, Reaction
from flux_weave.thermodynamics import (
    analyze_stoichiometric_kernel,
    compute_conserved_moieties,
    detect_futile_cycles,
)


def test_conserved_moiety_closed_cycle() -> None:
    # A <=> B <=> C with closed pool: A + B + C = const
    model = MetabolicModel("closed_pool")
    a = Metabolite("A")
    b = Metabolite("B")
    c = Metabolite("C")

    # R1: A -> B
    r1 = Reaction("R1", lower_bound=-10.0, upper_bound=10.0)
    r1.add_metabolite(a, -1.0)
    r1.add_metabolite(b, 1.0)

    # R2: B -> C
    r2 = Reaction("R2", lower_bound=-10.0, upper_bound=10.0)
    r2.add_metabolite(b, -1.0)
    r2.add_metabolite(c, 1.0)

    model.add_reactions([r1, r2])

    moieties = compute_conserved_moieties(model, internal_only=False)
    assert len(moieties) >= 1
    moiety = moieties[0]
    # Check that A, B, and C all have equal non-zero coefficients
    coeffs = moiety.metabolite_coefficients
    assert "A" in coeffs and "B" in coeffs and "C" in coeffs
    assert np.isclose(abs(coeffs["A"]), abs(coeffs["B"]))
    assert np.isclose(abs(coeffs["B"]), abs(coeffs["C"]))


def test_stoichiometric_kernel_analysis() -> None:
    model = create_toy_linear_model()
    report = analyze_stoichiometric_kernel(model)

    assert report.num_metabolites == 3
    assert report.num_reactions == 4
    # Rank of linear chain S is 3
    assert report.rank == 3
    assert report.degrees_of_freedom == 1


def test_futile_cycles_detection() -> None:
    model = create_toy_linear_model()
    # Initially no internal futile cycles
    cycles = detect_futile_cycles(model)
    assert len(cycles) == 0

    # Add futile cycle: X <=> Y
    x = Metabolite("X")
    y = Metabolite("Y")
    rf = Reaction("LOOP_FWD", lower_bound=0.0, upper_bound=10.0)
    rf.add_metabolite(x, -1.0)
    rf.add_metabolite(y, 1.0)
    rr = Reaction("LOOP_REV", lower_bound=0.0, upper_bound=10.0)
    rr.add_metabolite(y, -1.0)
    rr.add_metabolite(x, 1.0)
    model.add_reactions([rf, rr])

    cycles_with_loop = detect_futile_cycles(model)
    assert len(cycles_with_loop) >= 2
    rxn_ids = [c[0] for c in cycles_with_loop]
    assert "LOOP_FWD" in rxn_ids and "LOOP_REV" in rxn_ids
