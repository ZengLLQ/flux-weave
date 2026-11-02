"""Command Line Interface for FluxWeave.

Provides commands for FBA, FVA, pFBA, gene/reaction knockout screening,
and conserved moiety analysis.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from flux_weave.canonical import create_core_carbon_model, create_toy_linear_model
from flux_weave.fba import solve_fba
from flux_weave.fva import run_fva
from flux_weave.knockout import (
    double_gene_deletion,
    single_gene_deletion,
    single_reaction_deletion,
)
from flux_weave.model import MetabolicModel
from flux_weave.pfba import solve_pfba
from flux_weave.thermodynamics import analyze_stoichiometric_kernel


def get_model(name: str) -> MetabolicModel:
    """Retrieve canonical model by name."""
    if name == "toy":
        return create_toy_linear_model()
    return create_core_carbon_model()


def cmd_info(args: argparse.Namespace) -> int:
    """Display structural metadata of the model."""
    model = get_model(args.model)
    s_mat = model.stoichiometric_matrix()
    print("=" * 70)
    print(f" FluxWeave Metabolic Model: {model.id} ({model.name})")
    print("=" * 70)
    print(f" Metabolites:  {len(model.metabolites)}")
    print(f" Reactions:    {len(model.reactions)}")
    print(f" Genes:        {len(model.extract_all_genes())}")
    print(f" S-Matrix Dim: {s_mat.shape[0]} x {s_mat.shape[1]}")
    print("-" * 70)
    print(f"{'Reaction ID':<15} {'Subsystem':<20} {'Bounds':<18} {'Reaction Equation'}")
    print("-" * 70)
    for r in model.reactions:
        bounds_str = f"[{r.lower_bound:g}, {r.upper_bound:g}]"
        print(f"{r.id:<15} {r.subsystem[:18]:<20} {bounds_str:<18} {r.build_reaction_string()}")
    print("=" * 70)
    return 0


def cmd_fba(args: argparse.Namespace) -> int:
    """Run Flux Balance Analysis and report optimal fluxes and shadow prices."""
    model = get_model(args.model)
    sol = solve_fba(model, method=args.method)

    print("=" * 70)
    print(f" Flux Balance Analysis (FBA) — Model: {model.id}")
    print(f" Status: {sol.status.upper()} | Objective Value (Z*): {sol.objective_value:.6f}")
    print("=" * 70)

    print("\n[ Top Active Fluxes ]")
    print(f"{'Reaction ID':<15} {'Flux (mmol/gDW/h)':<20} {'Subsystem':<20}")
    print("-" * 70)
    for r_id, flux in sol.top_fluxes(n=args.top):
        rxn = model.get_reaction(r_id)
        print(f"{r_id:<15} {flux:<20.6f} {rxn.subsystem:<20}")

    print("\n[ Limiting Metabolite Shadow Prices (Dual Values) ]")
    print(f"{'Metabolite ID':<15} {'Shadow Price (dZ/db)':<25}")
    print("-" * 70)
    sorted_shadows = sorted(sol.shadow_prices.items(), key=lambda x: abs(x[1]), reverse=True)
    count = 0
    for m_id, sp in sorted_shadows:
        if abs(sp) > 1e-6 and count < args.top:
            print(f"{m_id:<15} {sp:<25.6f}")
            count += 1
    if count == 0:
        print(" None (no unconstrained bottleneck shadow prices)")
    print("=" * 70)
    return 0


def cmd_fva(args: argparse.Namespace) -> int:
    """Run Flux Variability Analysis and report flux elasticity intervals."""
    model = get_model(args.model)
    fva_res = run_fva(model, fraction_optimum=args.fraction)

    print("=" * 70)
    print(f" Flux Variability Analysis (FVA) — Model: {model.id}")
    target_str = f"Z* >= {args.fraction * fva_res.objective_value:.6f}"
    print(f" Target Objective Fraction (gamma): {args.fraction:.2f} ({target_str})")
    print("=" * 70)
    print(f"{'Reaction ID':<15} {'Min Flux':<12} {'Max Flux':<12} {'Span':<12} {'Classification'}")
    print("-" * 70)

    for r_id, res in fva_res.results.items():
        cls_str = "BLOCKED" if res.is_blocked else "ESSENTIAL" if res.is_essential else "PLASTIC"
        print(f"{r_id:<15} {res.minimum:<12.4f} {res.maximum:<12.4f} {res.span:<12.4f} {cls_str}")

    print("-" * 70)
    print(f" Essential Reactions: {len(fva_res.essential_reactions)}")
    print(f" Blocked Reactions:   {len(fva_res.blocked_reactions)}")
    print(f" Plastic Reactions:   {len(fva_res.plastic_reactions)}")
    print("=" * 70)
    return 0


def cmd_pfba(args: argparse.Namespace) -> int:
    """Run Parsimonious FBA and report total enzyme burden."""
    model = get_model(args.model)
    pfba_sol = solve_pfba(model, fraction_optimum=args.fraction)

    print("=" * 70)
    print(f" Parsimonious FBA (pFBA) — Model: {model.id}")
    print(f" Objective Value:       {pfba_sol.objective_value:.6f}")
    print(f" Total Enzyme Burden:   {pfba_sol.total_flux_burden:.6f} mmol/gDW/h")
    print("=" * 70)
    print(f"{'Reaction ID':<15} {'pFBA Flux':<18} {'Subsystem'}")
    print("-" * 70)
    active = pfba_sol.active_reactions()
    for r_id, flux in sorted(active, key=lambda x: abs(x[1]), reverse=True)[:args.top]:
        rxn = model.get_reaction(r_id)
        print(f"{r_id:<15} {flux:<18.6f} {rxn.subsystem}")
    print("=" * 70)
    return 0


def cmd_knockout(args: argparse.Namespace) -> int:
    """Screen single/double gene or reaction deletions."""
    model = get_model(args.model)

    print("=" * 70)
    print(f" In Silico Knockout Screening — Mode: {args.mode.upper()}")
    print("=" * 70)

    if args.mode == "reaction":
        del_res = single_reaction_deletion(model)
        print(f"{'Reaction ID':<15} {'KO Growth':<14} {'Fraction':<12} {'Essentiality'}")
        print("-" * 70)
        for r_id, res in del_res.items():
            cat = res.category.value.upper()
            growth_s = f"{res.knockout_growth:<14.6f}"
            print(f"{r_id:<15} {growth_s} {res.growth_fraction:<12.2%} {cat}")

    elif args.mode == "gene":
        gene_res = single_gene_deletion(model)
        print(f"{'Gene ID':<15} {'KO Growth':<14} {'Fraction':<12} {'Essentiality'}")
        print("-" * 70)
        for g_id, res in sorted(gene_res.items()):
            cat = res.category.value.upper()
            growth_s = f"{res.knockout_growth:<14.6f}"
            print(f"{g_id:<15} {growth_s} {res.growth_fraction:<12.2%} {cat}")

    elif args.mode == "synthetic":
        syn_res = double_gene_deletion(model)
        syn_lethals = [p for p, res in syn_res.items() if res.is_synthetic_lethal]
        print(f" Screened {len(syn_res)} non-essential gene pairs.")
        print(f" Found {len(syn_lethals)} Synthetic Lethal gene pairs:")
        print("-" * 70)
        print(f"{'Gene Pair':<25} {'KO Growth':<15} {'Status'}")
        print("-" * 70)
        for p in syn_lethals:
            res = syn_res[p]
            pair_str = f"({p[0]}, {p[1]})"
            print(f"{pair_str:<25} {res.knockout_growth:<15.6f} SYNTHETIC LETHAL")

    print("=" * 70)
    return 0


def cmd_moieties(args: argparse.Namespace) -> int:
    """Compute left null space and identify conserved chemical moieties."""
    model = get_model(args.model)
    report = analyze_stoichiometric_kernel(model)

    print("=" * 70)
    print(f" Stoichiometric Kernel & Conserved Moieties — Model: {model.id}")
    print("=" * 70)
    print(f" S Matrix Rank:         {report.rank}")
    print(f" Degrees of Freedom:    {report.degrees_of_freedom}")
    print(f" Conserved Moieties:    {len(report.conserved_moieties)}")
    print("-" * 70)
    for m in report.conserved_moieties:
        print(f" [{m.id}] Conservation Equation:")
        print(f"   d/dt ({m.formula_str()}) = 0")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line arguments parser."""
    parser = argparse.ArgumentParser(
        prog="flux-weave",
        description="High-Performance Flux Balance Analysis & Stoichiometry Engine.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display metabolic network architecture")
    p_info.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")

    # fba
    p_fba = subparsers.add_parser("fba", help="Execute Flux Balance Analysis")
    p_fba.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")
    p_fba.add_argument("--method", choices=["highs", "highs-ds", "highs-ipm"], default="highs")
    p_fba.add_argument("--top", type=int, default=10, help="Number of top reactions to display")

    # fva
    p_fva = subparsers.add_parser("fva", help="Execute Flux Variability Analysis")
    p_fva.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")
    p_fva.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of optimum [0.0, 1.0]",
    )

    # pfba
    p_pfba = subparsers.add_parser("pfba", help="Execute Parsimonious FBA")
    p_pfba.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")
    p_pfba.add_argument("--fraction", type=float, default=1.0, help="Fraction of optimum")
    p_pfba.add_argument("--top", type=int, default=10, help="Top reactions to display")

    # knockout
    p_ko = subparsers.add_parser("knockout", help="Screen genetic and reaction deletions")
    p_ko.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")
    p_ko.add_argument(
        "--mode",
        choices=["reaction", "gene", "synthetic"],
        default="reaction",
        help="Deletion mode",
    )

    # moieties
    p_moi = subparsers.add_parser("moieties", help="Analyze conserved stoichiometric pools")
    p_moi.add_argument("--model", choices=["core", "toy"], default="core", help="Model choice")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "info": cmd_info,
        "fba": cmd_fba,
        "fva": cmd_fva,
        "pfba": cmd_pfba,
        "knockout": cmd_knockout,
        "moieties": cmd_moieties,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
