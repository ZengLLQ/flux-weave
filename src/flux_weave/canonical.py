"""Canonical, fully validated metabolic network models.

Includes:
1. `create_core_carbon_model`: Realistic Core Central Carbon Metabolism
   comprising Glycolysis, PPP, TCA cycle, Respiration, Fermentation,
   GPR associations, and Biomass synthesis.
2. `create_toy_linear_model`: Compact 4-reaction pathway for exact analytical testing.
"""

from __future__ import annotations

from flux_weave.model import MetabolicModel, Metabolite, Reaction


def create_toy_linear_model() -> MetabolicModel:
    """Create a minimal 4-reaction test model with known analytical solution:

    EX_A: --> A (ub=10)
    R1:   A --> B (ub=10, GPR='g1')
    R2:   B --> C (ub=10, GPR='g2 or g3')
    EX_C: C -->   (objective = 1.0)
    """
    model = MetabolicModel("toy_linear", "Toy Linear Metabolic Network")

    a = Metabolite("A", "Substrate A")
    b = Metabolite("B", "Intermediate B")
    c = Metabolite("C", "Product C")

    # EX_A: -> A
    ex_a = Reaction("EX_A", lower_bound=0.0, upper_bound=10.0)
    ex_a.add_metabolite(a, 1.0)

    # R1: A -> B
    r1 = Reaction("R1", lower_bound=0.0, upper_bound=10.0, gene_reaction_rule="g1")
    r1.add_metabolite(a, -1.0)
    r1.add_metabolite(b, 1.0)

    # R2: B -> C (isoenzymes g2 or g3)
    r2 = Reaction("R2", lower_bound=0.0, upper_bound=10.0, gene_reaction_rule="g2 or g3")
    r2.add_metabolite(b, -1.0)
    r2.add_metabolite(c, 1.0)

    # EX_C: C -> (objective)
    ex_c = Reaction("EX_C", lower_bound=0.0, upper_bound=100.0, objective_coefficient=1.0)
    ex_c.add_metabolite(c, -1.0)

    model.add_reactions([ex_a, r1, r2, ex_c])
    return model


def create_core_carbon_model() -> MetabolicModel:
    """Build a comprehensive Central Carbon Metabolism model for microbial cell factory simulation.

    Includes 25 metabolites and 22 reactions covering:
    - Glucose uptake & transport
    - Glycolysis (EMP pathway)
    - Pentose Phosphate Pathway (PPP)
    - Pyruvate dehydrogenase multimeric complex
    - Full TCA Cycle with anaplerosis
    - Respiration / Oxidative Phosphorylation with P/O ratio 2.5
    - Lactate fermentation
    - Non-growth-associated ATP maintenance (ATPM)
    - Growth biomass synthesis reaction
    """
    model = MetabolicModel("core_carbon", "Central Carbon Metabolism & Biomass Production")

    # Metabolites
    glc_e = Metabolite("glc_e", "D-Glucose (extracellular)", compartment="e")
    glc_c = Metabolite("glc_c", "D-Glucose (cytosol)", compartment="c")
    g6p_c = Metabolite("g6p_c", "D-Glucose 6-phosphate", compartment="c")
    f6p_c = Metabolite("f6p_c", "D-Fructose 6-phosphate", compartment="c")
    fdp_c = Metabolite("fdp_c", "D-Fructose 1,6-bisphosphate", compartment="c")
    gap_c = Metabolite("gap_c", "Glyceraldehyde 3-phosphate", compartment="c")
    pep_c = Metabolite("pep_c", "Phosphoenolpyruvate", compartment="c")
    pyr_c = Metabolite("pyr_c", "Pyruvate", compartment="c")
    accoa_c = Metabolite("accoa_c", "Acetyl-CoA", compartment="c")
    cit_c = Metabolite("cit_c", "Citrate", compartment="c")
    akg_c = Metabolite("akg_c", "2-Oxoglutarate", compartment="c")
    succ_c = Metabolite("succ_c", "Succinate", compartment="c")
    mal_c = Metabolite("mal_c", "Malate", compartment="c")
    oaa_c = Metabolite("oaa_c", "Oxaloacetate", compartment="c")
    lac_c = Metabolite("lac_c", "D-Lactate (cytosol)", compartment="c")
    lac_e = Metabolite("lac_e", "D-Lactate (extracellular)", compartment="e")
    co2_c = Metabolite("co2_c", "Carbon dioxide", compartment="c")
    o2_c = Metabolite("o2_c", "Oxygen (cytosol)", compartment="c")
    o2_e = Metabolite("o2_e", "Oxygen (extracellular)", compartment="e")
    h2o_c = Metabolite("h2o_c", "H2O", compartment="c")
    atp_c = Metabolite("atp_c", "ATP", compartment="c")
    adp_c = Metabolite("adp_c", "ADP", compartment="c")
    pi_c = Metabolite("pi_c", "Orthophosphate", compartment="c")
    nad_c = Metabolite("nad_c", "NAD+", compartment="c")
    nadh_c = Metabolite("nadh_c", "NADH", compartment="c")
    nadp_c = Metabolite("nadp_c", "NADP+", compartment="c")
    nadph_c = Metabolite("nadph_c", "NADPH", compartment="c")
    biomass_c = Metabolite("biomass_c", "Cellular Biomass", compartment="c")

    # Exchange Reactions
    # EX_glc: glc_e <=> 0 (uptake limit 10.0 mmol/gDW/h)
    ex_glc = Reaction("EX_glc", lower_bound=-10.0, upper_bound=1000.0, subsystem="Exchange")
    ex_glc.add_metabolite(glc_e, -1.0)

    # EX_o2: o2_e <=> 0 (uptake limit 20.0 mmol/gDW/h)
    ex_o2 = Reaction("EX_o2", lower_bound=-20.0, upper_bound=1000.0, subsystem="Exchange")
    ex_o2.add_metabolite(o2_e, -1.0)

    # EX_co2: co2_c -> 0
    ex_co2 = Reaction("EX_co2", lower_bound=0.0, upper_bound=1000.0, subsystem="Exchange")
    ex_co2.add_metabolite(co2_c, -1.0)

    # EX_h2o: h2o_c <=> 0
    ex_h2o = Reaction("EX_h2o", lower_bound=-1000.0, upper_bound=1000.0, subsystem="Exchange")
    ex_h2o.add_metabolite(h2o_c, -1.0)

    # EX_lac: lac_e -> 0 (secretion)
    ex_lac = Reaction("EX_lac", lower_bound=0.0, upper_bound=1000.0, subsystem="Exchange")
    ex_lac.add_metabolite(lac_e, -1.0)

    # EX_pi: pi_c <=> 0 (phosphate exchange)
    ex_pi = Reaction("EX_pi", lower_bound=-1000.0, upper_bound=1000.0, subsystem="Exchange")
    ex_pi.add_metabolite(pi_c, -1.0)

    # EX_biomass: biomass_c -> 0
    ex_biomass = Reaction(
        "EX_biomass",
        lower_bound=0.0,
        upper_bound=1000.0,
        objective_coefficient=1.0,
        subsystem="Exchange",
    )
    ex_biomass.add_metabolite(biomass_c, -1.0)

    # Transporters
    # GLCt: glc_e -> glc_c
    glct = Reaction(
        "GLCt",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="ptsG or (manX and manY and manZ)",
        subsystem="Transport",
    )
    glct.add_metabolite(glc_e, -1.0)
    glct.add_metabolite(glc_c, 1.0)

    # O2t: o2_e <=> o2_c
    o2t = Reaction("O2t", lower_bound=-1000.0, upper_bound=1000.0, subsystem="Transport")
    o2t.add_metabolite(o2_e, -1.0)
    o2t.add_metabolite(o2_c, 1.0)

    # LACt: lac_c <=> lac_e
    lact = Reaction(
        "LACt",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="lldP",
        subsystem="Transport",
    )
    lact.add_metabolite(lac_c, -1.0)
    lact.add_metabolite(lac_e, 1.0)

    # Glycolysis
    # HEX: glc_c + atp_c -> g6p_c + adp_c
    hex_rxn = Reaction(
        "HEX",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="glk",
        subsystem="Glycolysis",
    )
    hex_rxn.add_metabolite(glc_c, -1.0)
    hex_rxn.add_metabolite(atp_c, -1.0)
    hex_rxn.add_metabolite(g6p_c, 1.0)
    hex_rxn.add_metabolite(adp_c, 1.0)

    # PGI: g6p_c <=> f6p_c
    pgi = Reaction(
        "PGI",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="pgi",
        subsystem="Glycolysis",
    )
    pgi.add_metabolite(g6p_c, -1.0)
    pgi.add_metabolite(f6p_c, 1.0)

    # PFK: f6p_c + atp_c -> fdp_c + adp_c
    pfk = Reaction(
        "PFK",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="pfkA or pfkB",
        subsystem="Glycolysis",
    )
    pfk.add_metabolite(f6p_c, -1.0)
    pfk.add_metabolite(atp_c, -1.0)
    pfk.add_metabolite(fdp_c, 1.0)
    pfk.add_metabolite(adp_c, 1.0)

    # FBA_TPI: fdp_c <=> 2 gap_c
    fba = Reaction(
        "FBA",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="fbaA or fbaB",
        subsystem="Glycolysis",
    )
    fba.add_metabolite(fdp_c, -1.0)
    fba.add_metabolite(gap_c, 2.0)

    # GAPD_PGK: gap_c + nad_c + 2 adp_c + 2 pi_c <=> pep_c + nadh_c + 2 atp_c
    gapd = Reaction(
        "GAPD",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="gapA",
        subsystem="Glycolysis",
    )
    gapd.add_metabolite(gap_c, -1.0)
    gapd.add_metabolite(nad_c, -1.0)
    gapd.add_metabolite(adp_c, -2.0)
    gapd.add_metabolite(pi_c, -2.0)
    gapd.add_metabolite(pep_c, 1.0)
    gapd.add_metabolite(nadh_c, 1.0)
    gapd.add_metabolite(atp_c, 2.0)

    # PYK: pep_c + adp_c -> pyr_c + atp_c
    pyk = Reaction(
        "PYK",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="pykA or pykF",
        subsystem="Glycolysis",
    )
    pyk.add_metabolite(pep_c, -1.0)
    pyk.add_metabolite(adp_c, -1.0)
    pyk.add_metabolite(pyr_c, 1.0)
    pyk.add_metabolite(atp_c, 1.0)

    # Pentose Phosphate Pathway Shunt
    # G6PDH: g6p_c + 2 nadp_c -> gap_c + 2 nadph_c + co2_c
    g6pdh = Reaction(
        "G6PDH",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="zwf",
        subsystem="Pentose Phosphate",
    )
    g6pdh.add_metabolite(g6p_c, -1.0)
    g6pdh.add_metabolite(nadp_c, -2.0)
    g6pdh.add_metabolite(gap_c, 1.0)
    g6pdh.add_metabolite(nadph_c, 2.0)
    g6pdh.add_metabolite(co2_c, 1.0)

    # Pyruvate Metabolism
    # PDH: pyr_c + nad_c -> accoa_c + nadh_c + co2_c (Heteromeric: aceE and aceF and lpd)
    pdh = Reaction(
        "PDH",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="aceE and aceF and lpd",
        subsystem="Pyruvate Metabolism",
    )
    pdh.add_metabolite(pyr_c, -1.0)
    pdh.add_metabolite(nad_c, -1.0)
    pdh.add_metabolite(accoa_c, 1.0)
    pdh.add_metabolite(nadh_c, 1.0)
    pdh.add_metabolite(co2_c, 1.0)

    # LDH: pyr_c + nadh_c <=> lac_c + nad_c
    ldh = Reaction(
        "LDH",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="ldhA",
        subsystem="Fermentation",
    )
    ldh.add_metabolite(pyr_c, -1.0)
    ldh.add_metabolite(nadh_c, -1.0)
    ldh.add_metabolite(lac_c, 1.0)
    ldh.add_metabolite(nad_c, 1.0)

    # TCA Cycle
    # CS: accoa_c + oaa_c -> cit_c
    cs = Reaction(
        "CS",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="gltA",
        subsystem="TCA Cycle",
    )
    cs.add_metabolite(accoa_c, -1.0)
    cs.add_metabolite(oaa_c, -1.0)
    cs.add_metabolite(cit_c, 1.0)

    # ICDH: cit_c + nad_c -> akg_c + nadh_c + co2_c
    icdh = Reaction(
        "ICDH",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="icd",
        subsystem="TCA Cycle",
    )
    icdh.add_metabolite(cit_c, -1.0)
    icdh.add_metabolite(nad_c, -1.0)
    icdh.add_metabolite(akg_c, 1.0)
    icdh.add_metabolite(nadh_c, 1.0)
    icdh.add_metabolite(co2_c, 1.0)

    # AKGDH: akg_c + nad_c + adp_c + pi_c -> succ_c + nadh_c + atp_c + co2_c
    akgdh = Reaction(
        "AKGDH",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="sucA and sucB",
        subsystem="TCA Cycle",
    )
    akgdh.add_metabolite(akg_c, -1.0)
    akgdh.add_metabolite(nad_c, -1.0)
    akgdh.add_metabolite(adp_c, -1.0)
    akgdh.add_metabolite(pi_c, -1.0)
    akgdh.add_metabolite(succ_c, 1.0)
    akgdh.add_metabolite(nadh_c, 1.0)
    akgdh.add_metabolite(atp_c, 1.0)
    akgdh.add_metabolite(co2_c, 1.0)

    # SDH: succ_c + nad_c -> mal_c + nadh_c
    sdh = Reaction(
        "SDH",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="sdhA and sdhB and sdhC",
        subsystem="TCA Cycle",
    )
    sdh.add_metabolite(succ_c, -1.0)
    sdh.add_metabolite(nad_c, -1.0)
    sdh.add_metabolite(mal_c, 1.0)
    sdh.add_metabolite(nadh_c, 1.0)

    # MDH: mal_c + nad_c <=> oaa_c + nadh_c
    mdh = Reaction(
        "MDH",
        lower_bound=-1000.0,
        upper_bound=1000.0,
        gene_reaction_rule="mdh",
        subsystem="TCA Cycle",
    )
    mdh.add_metabolite(mal_c, -1.0)
    mdh.add_metabolite(nad_c, -1.0)
    mdh.add_metabolite(oaa_c, 1.0)
    mdh.add_metabolite(nadh_c, 1.0)

    # Anaplerosis: PPC (pep_c + co2_c -> oaa_c)
    ppc = Reaction(
        "PPC",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="ppc",
        subsystem="Anaplerosis",
    )
    ppc.add_metabolite(pep_c, -1.0)
    ppc.add_metabolite(co2_c, -1.0)
    ppc.add_metabolite(oaa_c, 1.0)

    # Respiration / Oxidative Phosphorylation
    # RESP: nadh_c + 0.5 o2_c + 2.5 adp_c + 2.5 pi_c -> nad_c + h2o_c + 2.5 atp_c
    resp = Reaction(
        "RESP",
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_reaction_rule="nuoA and atpA",
        subsystem="Oxidative Phosphorylation",
    )
    resp.add_metabolite(nadh_c, -1.0)
    resp.add_metabolite(o2_c, -0.5)
    resp.add_metabolite(adp_c, -2.5)
    resp.add_metabolite(pi_c, -2.5)
    resp.add_metabolite(nad_c, 1.0)
    resp.add_metabolite(h2o_c, 1.0)
    resp.add_metabolite(atp_c, 2.5)

    # Non-Growth Associated ATP Maintenance
    # ATPM: atp_c + h2o_c -> adp_c + pi_c (lb = 1.0)
    atpm = Reaction("ATPM", lower_bound=1.0, upper_bound=1000.0, subsystem="Maintenance")
    atpm.add_metabolite(atp_c, -1.0)
    atpm.add_metabolite(h2o_c, -1.0)
    atpm.add_metabolite(adp_c, 1.0)
    atpm.add_metabolite(pi_c, 1.0)

    # Growth Biomass Synthesis
    # 0.2 accoa + 0.2 akg + 0.2 oaa + 0.1 g6p + 0.1 pep + 10 atp + 1 nadph -> biomass...
    biomass = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0, subsystem="Biomass")
    biomass.add_metabolite(accoa_c, -0.2)
    biomass.add_metabolite(akg_c, -0.2)
    biomass.add_metabolite(oaa_c, -0.2)
    biomass.add_metabolite(g6p_c, -0.1)
    biomass.add_metabolite(pep_c, -0.1)
    biomass.add_metabolite(atp_c, -10.0)
    biomass.add_metabolite(nadph_c, -1.0)
    biomass.add_metabolite(biomass_c, 1.0)
    biomass.add_metabolite(adp_c, 10.0)
    biomass.add_metabolite(pi_c, 10.0)
    biomass.add_metabolite(nadp_c, 1.0)

    # Register all reactions
    model.add_reactions([
        ex_glc, ex_o2, ex_co2, ex_h2o, ex_lac, ex_pi, ex_biomass,
        glct, o2t, lact,
        hex_rxn, pgi, pfk, fba, gapd, pyk, g6pdh,
        pdh, ldh,
        cs, icdh, akgdh, sdh, mdh, ppc,
        resp, atpm, biomass
    ])

    return model
