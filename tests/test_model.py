"""Unit tests for metabolic data models and stoichiometric matrix generation."""

import pytest
from flux_weave.model import Gene, MetabolicModel, Metabolite, Reaction


def test_metabolite_properties() -> None:
    m = Metabolite(id="glc_c", name="D-Glucose", compartment="c", formula="C6H12O6", charge=0)
    assert m.id == "glc_c"
    assert m.compartment == "c"
    assert "glc_c" in repr(m)

    with pytest.raises(ValueError, match="Metabolite ID cannot be empty"):
        Metabolite(id="")


def test_reaction_bounds_validation() -> None:
    with pytest.raises(ValueError, match="lower bound .* exceeds upper bound"):
        Reaction(id="RXN_BAD", lower_bound=10.0, upper_bound=5.0)

    rxn = Reaction(id="R1", lower_bound=-10.0, upper_bound=10.0)
    assert rxn.is_reversible is True

    rxn_irrev = Reaction(id="R2", lower_bound=0.0, upper_bound=100.0)
    assert rxn_irrev.is_reversible is False


def test_reaction_stoichiometry_and_string() -> None:
    m1 = Metabolite(id="A")
    m2 = Metabolite(id="B")
    m3 = Metabolite(id="C")

    rxn = Reaction(id="R_TEST", lower_bound=0.0, upper_bound=50.0)
    rxn.add_metabolite(m1, -1.0)
    rxn.add_metabolite(m2, -2.0)
    rxn.add_metabolite(m3, 3.0)

    assert rxn.reactants == {m1: 1.0, m2: 2.0}
    assert rxn.products == {m3: 3.0}

    eq_str = rxn.build_reaction_string()
    assert "A" in eq_str and "2 B" in eq_str and "3 C" in eq_str
    assert "-->" in eq_str

    # Test removing metabolite by setting coefficient to 0
    rxn.add_metabolite(m2, 0.0)
    assert m2 not in rxn.metabolites


def test_metabolic_model_stoichiometry() -> None:
    model = MetabolicModel("test_model", "Test Model")
    m1 = Metabolite("M1")
    m2 = Metabolite("M2")
    model.add_metabolites([m1, m2])

    r1 = Reaction("R1", lower_bound=0.0, upper_bound=10.0)
    r1.add_metabolite(m1, -1.0)
    r1.add_metabolite(m2, 1.0)
    model.add_reaction(r1)

    s_mat = model.stoichiometric_matrix()
    assert s_mat.shape == (2, 1)
    assert s_mat[0, 0] == -1.0
    assert s_mat[1, 0] == 1.0


def test_model_json_serialization_roundtrip() -> None:
    model = MetabolicModel("ser_model", "Serialization Test")
    m1 = Metabolite("X", compartment="c", formula="H2O")
    m2 = Metabolite("Y", compartment="c")
    g = Gene("g1", "Gene 1")
    model.add_metabolite(m1)
    model.add_metabolite(m2)
    model.add_gene(g)

    r = Reaction(
        "R_SER",
        lower_bound=-5.0,
        upper_bound=20.0,
        objective_coefficient=1.0,
        gene_reaction_rule="g1",
    )
    r.add_metabolite(m1, -1.0)
    r.add_metabolite(m2, 2.0)
    model.add_reaction(r)

    json_str = model.to_json()
    reloaded = MetabolicModel.from_json(json_str)

    assert reloaded.id == "ser_model"
    assert len(reloaded.metabolites) == 2
    assert len(reloaded.reactions) == 1
    assert len(reloaded.genes) == 1
    rxn_reloaded = reloaded.get_reaction("R_SER")
    assert rxn_reloaded.lower_bound == -5.0
    assert rxn_reloaded.upper_bound == 20.0
    assert rxn_reloaded.objective_coefficient == 1.0
    assert rxn_reloaded.gene_reaction_rule == "g1"


def test_model_deep_copy() -> None:
    model = MetabolicModel("m_orig")
    m = Metabolite("A")
    r = Reaction("R1", lower_bound=0.0, upper_bound=10.0)
    r.add_metabolite(m, 1.0)
    model.add_reaction(r)

    copied = model.copy()
    assert copied.id == model.id
    # Modify copy and ensure original remains unaffected
    copied.get_reaction("R1").upper_bound = 999.0
    assert model.get_reaction("R1").upper_bound == 10.0
