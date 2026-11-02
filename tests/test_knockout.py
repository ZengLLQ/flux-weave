"""Unit tests for single and double gene/reaction knockout screening and synthetic lethality."""

import pytest
from flux_weave.canonical import create_core_carbon_model, create_toy_linear_model
from flux_weave.knockout import (
    EssentialityCategory,
    double_gene_deletion,
    single_gene_deletion,
    single_reaction_deletion,
)


def test_single_reaction_deletion_toy() -> None:
    model = create_toy_linear_model()
    res = single_reaction_deletion(model)

    # Every reaction in a linear chain is essential
    for r_id in ["EX_A", "R1", "R2", "EX_C"]:
        assert res[r_id].category == EssentialityCategory.ESSENTIAL
        assert pytest.approx(res[r_id].knockout_growth, abs=1e-5) == 0.0


def test_single_gene_deletion_toy() -> None:
    # R1: g1 (essential)
    # R2: g2 or g3 (isoenzymes: g2 alone is non-essential, g3 alone is non-essential)
    model = create_toy_linear_model()
    res = single_gene_deletion(model)

    assert res["g1"].category == EssentialityCategory.ESSENTIAL
    assert pytest.approx(res["g1"].knockout_growth, abs=1e-5) == 0.0

    assert res["g2"].category == EssentialityCategory.NON_ESSENTIAL
    assert pytest.approx(res["g2"].knockout_growth, rel=1e-5) == 10.0

    assert res["g3"].category == EssentialityCategory.NON_ESSENTIAL
    assert pytest.approx(res["g3"].knockout_growth, rel=1e-5) == 10.0


def test_synthetic_lethality_detection() -> None:
    # Double deletion of (g2, g3) knocks out R2 and halts growth -> SYNTHETIC LETHAL
    model = create_toy_linear_model()
    double_res = double_gene_deletion(model, gene_pairs=[("g2", "g3")])

    pair_result = double_res[("g2", "g3")]
    assert pair_result.is_synthetic_lethal is True
    assert pair_result.category == EssentialityCategory.ESSENTIAL
    assert pytest.approx(pair_result.knockout_growth, abs=1e-5) == 0.0


def test_gene_deletion_core_carbon() -> None:
    model = create_core_carbon_model()
    # Test hexokinase knockout (glk)
    res = single_gene_deletion(model, gene_ids=["glk"])
    assert "glk" in res
    assert res["glk"].wild_type_growth > 0.0
