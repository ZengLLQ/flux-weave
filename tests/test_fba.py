"""Unit tests for Flux Balance Analysis (FBA) solver and dual variables."""

import pytest
from flux_weave.canonical import create_core_carbon_model, create_toy_linear_model
from flux_weave.fba import solve_fba
from flux_weave.model import MetabolicModel, Metabolite, Reaction


def test_fba_toy_linear_exact_solution() -> None:
    # In toy model: EX_A (<= 10) -> R1 -> R2 -> EX_C (obj=1)
    # The analytical maximum flux must be exactly 10.0
    model = create_toy_linear_model()
    sol = solve_fba(model)

    assert sol.is_optimal is True
    assert pytest.approx(sol.objective_value, rel=1e-5) == 10.0
    assert pytest.approx(sol.get_flux("EX_A"), rel=1e-5) == 10.0
    assert pytest.approx(sol.get_flux("R1"), rel=1e-5) == 10.0
    assert pytest.approx(sol.get_flux("R2"), rel=1e-5) == 10.0
    assert pytest.approx(sol.get_flux("EX_C"), rel=1e-5) == 10.0


def test_fba_core_carbon_metabolism() -> None:
    model = create_core_carbon_model()
    sol = solve_fba(model)

    assert sol.is_optimal is True
    # Ensure viable cellular growth rate
    assert sol.objective_value > 0.05
    assert sol.objective_value < 50.0

    # Glucose uptake must not exceed -10.0 (lower bound)
    glc_flux = sol.get_flux("EX_glc")
    assert glc_flux >= -10.0001
    assert glc_flux < 0.0  # Must consume glucose

    # Respiration / Oxygen uptake
    o2_flux = sol.get_flux("EX_o2")
    assert o2_flux >= -20.0001
    assert o2_flux < 0.0

    # Shadow prices should be defined
    assert "glc_c" in sol.shadow_prices
    assert "atp_c" in sol.shadow_prices


def test_fba_infeasible_network() -> None:
    # Model requiring output with zero input
    model = MetabolicModel("infeasible")
    m = Metabolite("P")
    r_sink = Reaction("SINK", lower_bound=10.0, upper_bound=10.0, objective_coefficient=1.0)
    r_sink.add_metabolite(m, -1.0)
    model.add_reaction(r_sink)

    sol = solve_fba(model)
    assert sol.is_optimal is False
    assert sol.status in ("infeasible", "failed_2")
