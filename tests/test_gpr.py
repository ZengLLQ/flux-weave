"""Unit tests for Gene-Protein-Reaction (GPR) AST parsing and Boolean evaluation."""

from flux_weave.gpr import (
    AndNode,
    GeneNode,
    OrNode,
    eval_gpr,
    extract_genes_from_rule,
    parse_gpr,
)


def test_gpr_single_gene() -> None:
    ast = parse_gpr("b0116")
    assert isinstance(ast, GeneNode)
    assert ast.gene_id == "b0116"
    assert extract_genes_from_rule("b0116") == {"b0116"}

    # Knockout evaluation
    assert eval_gpr("b0116", {"b0116"}, is_knockout_set=True) is False
    assert eval_gpr("b0116", {"b9999"}, is_knockout_set=True) is True


def test_gpr_and_logic() -> None:
    # Heteromeric complex: both subunits required
    rule = "geneA and geneB"
    ast = parse_gpr(rule)
    assert isinstance(ast, AndNode)
    assert extract_genes_from_rule(rule) == {"geneA", "geneB"}

    # Neither knocked out -> active
    assert eval_gpr(rule, set(), is_knockout_set=True) is True
    # Knock out geneA -> inactive
    assert eval_gpr(rule, {"geneA"}, is_knockout_set=True) is False
    # Knock out geneB -> inactive
    assert eval_gpr(rule, {"geneB"}, is_knockout_set=True) is False
    # Knock out both -> inactive
    assert eval_gpr(rule, {"geneA", "geneB"}, is_knockout_set=True) is False


def test_gpr_or_logic() -> None:
    # Isoenzymes: either subunit sufficient
    rule = "iso1 or iso2"
    ast = parse_gpr(rule)
    assert isinstance(ast, OrNode)
    assert extract_genes_from_rule(rule) == {"iso1", "iso2"}

    # Neither knocked out -> active
    assert eval_gpr(rule, set(), is_knockout_set=True) is True
    # Knock out iso1 -> still active via iso2
    assert eval_gpr(rule, {"iso1"}, is_knockout_set=True) is True
    # Knock out iso2 -> still active via iso1
    assert eval_gpr(rule, {"iso2"}, is_knockout_set=True) is True
    # Knock out both -> inactive
    assert eval_gpr(rule, {"iso1", "iso2"}, is_knockout_set=True) is False


def test_gpr_nested_parentheses() -> None:
    # Complex rule: (a and b) or (c and d)
    rule = "(subunit1 and subunit2) or (alt1 and alt2)"
    genes = extract_genes_from_rule(rule)
    assert genes == {"subunit1", "subunit2", "alt1", "alt2"}

    # Knock out subunit1 -> inactive for primary, but alt complex still intact -> active
    assert eval_gpr(rule, {"subunit1"}, is_knockout_set=True) is True
    # Knock out subunit1 and alt2 -> both complexes broken -> inactive
    assert eval_gpr(rule, {"subunit1", "alt2"}, is_knockout_set=True) is False


def test_gpr_empty_or_whitespace() -> None:
    assert extract_genes_from_rule("") == set()
    assert eval_gpr("", {"any_gene"}, is_knockout_set=True) is True
    assert eval_gpr("   ", set(), is_knockout_set=True) is True
