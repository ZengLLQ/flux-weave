"""Unit tests for Parsimonious FBA (pFBA) and futile cycle minimization."""

import pytest
from flux_weave.canonical import create_core_carbon_model, create_toy_linear_model
from flux_weave.fba import solve_fba
from flux_weave.model import Metabolite, Reaction
from flux_weave.pfba import solve_pfba


def test_pfba_toy_linear_burden() -> None:
    # 4 reactions carrying 10.0 flux each -> total burden = 40.0
    model = create_toy_linear_model()
    pfba = solve_pfba(model)

    assert pfba.is_optimal is True
    assert pytest.approx(pfba.objective_value, rel=1e-5) == 10.0
    assert pytest.approx(pfba.total_flux_burden, rel=1e-4) == 40.0


def test_pfba_futile_cycle_elimination() -> None:
    # Build toy model with an artificial futile loop: X <=> Y
    model = create_toy_linear_model()

    x = Metabolite("X")
    y = Metabolite("Y")

    # Forward: X -> Y
    rf = Reaction("LOOP_FWD", lower_bound=0.0, upper_bound=100.0)
    rf.add_metabolite(x, -1.0)
    rf.add_metabolite(y, 1.0)

    # Reverse: Y -> X
    rr = Reaction("LOOP_REV", lower_bound=0.0, upper_bound=100.0)
    rr.add_metabolite(y, -1.0)
    rr.add_metabolite(x, 1.0)

    model.add_reactions([rf, rr])

    # Run pFBA: The loop reactions must be driven to strictly 0.0
    pfba = solve_pfba(model)
    assert pfba.is_optimal is True
    assert pytest.approx(pfba.get_flux("LOOP_FWD"), abs=1e-5) == 0.0
    assert pytest.approx(pfba.get_flux("LOOP_REV"), abs=1e-5) == 0.0
    assert pytest.approx(pfba.total_flux_burden, rel=1e-4) == 40.0


def test_pfba_core_carbon_metabolism() -> None:
    model = create_core_carbon_model()
    baseline = solve_fba(model)
    pfba = solve_pfba(model)

    assert pfba.is_optimal is True
    # pFBA maintains exactly the same objective value
    assert pytest.approx(pfba.objective_value, rel=1e-5) == baseline.objective_value
    # Total flux burden is positive and finite
    assert pfba.total_flux_burden > 0.0
