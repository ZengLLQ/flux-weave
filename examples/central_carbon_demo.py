"""End-to-end simulation example demonstrating FluxWeave on Central Carbon Metabolism.

Performs:
1. Baseline FBA growth rate & flux distribution.
2. Parsimonious FBA (pFBA) to determine minimal enzyme burden.
3. Flux Variability Analysis (FVA) defining pathway plasticity.
4. Essential gene identification via Boolean GPR evaluation.
"""

from flux_weave.canonical import create_core_carbon_model
from flux_weave.fba import solve_fba
from flux_weave.fva import run_fva
from flux_weave.knockout import EssentialityCategory, single_gene_deletion
from flux_weave.pfba import solve_pfba
from flux_weave.thermodynamics import analyze_stoichiometric_kernel


def main() -> None:
    print("=" * 75)
    print(" FluxWeave: Central Carbon Metabolism Demonstration")
    print("=" * 75)

    model = create_core_carbon_model()
    print(
        f"Loaded '{model.id}' with {len(model.metabolites)} metabolites "
        f"and {len(model.reactions)} reactions."
    )

    # 1. Baseline FBA
    fba_sol = solve_fba(model)
    print(f"\n[1] FBA Status: {fba_sol.status.upper()}")
    print(f"    Optimal Biomass Synthesis Rate: {fba_sol.objective_value:.4f} hr^-1")
    print(f"    Glucose Uptake Rate (EX_glc):    {fba_sol.get_flux('EX_glc'):.4f} mmol/gDW/hr")
    print(f"    Oxygen Uptake Rate (EX_o2):      {fba_sol.get_flux('EX_o2'):.4f} mmol/gDW/hr")

    # 2. pFBA
    pfba_sol = solve_pfba(model)
    print("\n[2] Parsimonious FBA (pFBA):")
    print(f"    Total Enzyme Flux Burden:        {pfba_sol.total_flux_burden:.4f} mmol/gDW/hr")
    active_count = len(pfba_sol.active_reactions())
    total_rxns = len(model.reactions)
    print(f"    Active Reactions in Solution:    {active_count} / {total_rxns}")

    # 3. FVA
    fva_sol = run_fva(model, fraction_optimum=0.95)
    print("\n[3] Flux Variability Analysis (95% Optimum):")
    print(f"    Essential Reactions:             {len(fva_sol.essential_reactions)}")
    print(f"    Plastic (Flexible) Reactions:    {len(fva_sol.plastic_reactions)}")
    print(f"    Blocked Reactions:               {len(fva_sol.blocked_reactions)}")

    # 4. Gene Essentiality
    gene_del = single_gene_deletion(model)
    essential_genes = [
        g for g, res in gene_del.items() if res.category == EssentialityCategory.ESSENTIAL
    ]
    print("\n[4] In Silico Gene Essentiality:")
    print(f"    Screened {len(gene_del)} metabolic genes.")
    print(f"    Essential Genes Identified:      {', '.join(sorted(essential_genes))}")

    # 5. Stoichiometric Kernel
    kernel = analyze_stoichiometric_kernel(model)
    print("\n[5] Stoichiometric Kernel Decomposition:")
    print(f"    Stoichiometric Matrix Rank:      {kernel.rank}")
    print(f"    Degrees of Freedom:              {kernel.degrees_of_freedom}")
    print(f"    Conserved Chemical Moieties:     {len(kernel.conserved_moieties)}")
    for m in kernel.conserved_moieties:
        print(f"      - {m.id}: {m.formula_str()} = const")

    print("\n" + "=" * 75)
    print(" Simulation completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
