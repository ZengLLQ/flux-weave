"""Unit tests for FluxWeave command-line interface execution."""

from flux_weave.cli import main


def test_cli_info() -> None:
    exit_code = main(["info", "--model", "toy"])
    assert exit_code == 0

    exit_code_core = main(["info", "--model", "core"])
    assert exit_code_core == 0


def test_cli_fba() -> None:
    exit_code = main(["fba", "--model", "toy"])
    assert exit_code == 0

    exit_code_core = main(["fba", "--model", "core", "--top", "5"])
    assert exit_code_core == 0


def test_cli_fva() -> None:
    exit_code = main(["fva", "--model", "toy", "--fraction", "1.0"])
    assert exit_code == 0


def test_cli_pfba() -> None:
    exit_code = main(["pfba", "--model", "toy"])
    assert exit_code == 0


def test_cli_knockout() -> None:
    exit_code_rxn = main(["knockout", "--model", "toy", "--mode", "reaction"])
    assert exit_code_rxn == 0

    exit_code_gene = main(["knockout", "--model", "toy", "--mode", "gene"])
    assert exit_code_gene == 0

    exit_code_syn = main(["knockout", "--model", "toy", "--mode", "synthetic"])
    assert exit_code_syn == 0


def test_cli_moieties() -> None:
    exit_code = main(["moieties", "--model", "toy"])
    assert exit_code == 0


def test_cli_no_args() -> None:
    exit_code = main([])
    assert exit_code == 0
