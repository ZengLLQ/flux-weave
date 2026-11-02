"""Unit tests for Flux Variability Analysis (FVA)."""

import pytest
from flux_weave.canonical import create_core_carbon_model, create_toy_linear_model
from flux_weave.fva import run_fva


def test_fva_toy_linear_rigid() -> None:
    # In strict linear network at 100% optimum, every reaction has span = 0
    model = create_toy_linear_model()
    fva = run_fva(model, fraction_optimum=1.0)

    assert fva.objective_value == pytest.approx(10.0, rel=1e-5)
    for r_id in ["EX_A", "R1", "R2", "EX_C"]:
        res = fva.results[r_id]
        assert pytest.approx(res.minimum, abs=1e-4) == 10.0
        assert pytest.approx(res.maximum, abs=1e-4) == 10.0
        assert pytest.approx(res.span, abs=1e-4) == 0.0
        assert res.is_essential is True

    assert len(fva.essential_reactions) == 4
    assert len(fva.plastic_reactions) == 0


def test_fva_relaxed_optimum_expansion() -> None:
    # At gamma = 0.5, the minimum flux through R1 drops to 5.0 while max remains 10.0
    model = create_toy_linear_model()
    fva_half = run_fva(model, fraction_optimum=0.5)

    r1_res = fva_half.results["R1"]
    assert pytest.approx(r1_res.minimum, abs=1e-4) == 5.0
    assert pytest.approx(r1_res.maximum, abs=1e-4) == 10.0
    assert pytest.approx(r1_res.span, abs=1e-4) == 5.0
    assert len(fva_half.plastic_reactions) == 4


def test_fva_core_carbon_metabolism() -> None:
    model = create_core_carbon_model()
    fva = run_fva(model, fraction_optimum=1.0)

    # Biomass exchange must be essential
    assert "EX_biomass" in fva.essential_reactions
    # Glycolytic flux must be essential
    assert "HEX" in fva.essential_reactions
    assert len(fva.results) == len(model.reactions)
