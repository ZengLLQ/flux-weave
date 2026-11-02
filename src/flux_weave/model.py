"""Core metabolic data models: Metabolite, Gene, Reaction, and MetabolicModel.

Provides strict stoichiometric matrix construction, reaction bounds validation,
and model serialization for Genome-Scale Metabolic Network reconstructions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class Metabolite:
    """Metabolite chemical entity in a cellular compartment."""

    id: str
    name: str = ""
    compartment: str = "c"
    formula: Optional[str] = None
    charge: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Metabolite ID cannot be empty.")

    def __repr__(self) -> str:
        return f"Metabolite(id='{self.id}', compartment='{self.compartment}')"


@dataclass(frozen=True)
class Gene:
    """Gene entity associated with metabolic enzyme catalysts."""

    id: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Gene ID cannot be empty.")

    def __repr__(self) -> str:
        return f"Gene(id='{self.id}')"


@dataclass
class Reaction:
    """Biochemical or transport transformation between metabolites."""

    id: str
    name: str = ""
    metabolites: Dict[Metabolite, float] = field(default_factory=dict)
    lower_bound: float = 0.0
    upper_bound: float = 1000.0
    objective_coefficient: float = 0.0
    gene_reaction_rule: str = ""
    subsystem: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Reaction ID cannot be empty.")
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Reaction '{self.id}' lower bound ({self.lower_bound}) "
                f"exceeds upper bound ({self.upper_bound})."
            )

    @property
    def is_reversible(self) -> bool:
        """Return True if flux can flow in both forward and reverse directions."""
        return self.lower_bound < 0.0 and self.upper_bound > 0.0

    @property
    def is_exchange(self) -> bool:
        """Return True if this reaction acts as an exchange or boundary influx/efflux."""
        return len(self.metabolites) == 1 or self.id.startswith("EX_")

    def add_metabolite(self, metabolite: Metabolite, coefficient: float) -> None:
        """Add or update a metabolite's stoichiometric coefficient."""
        if coefficient == 0.0:
            self.metabolites.pop(metabolite, None)
        else:
            self.metabolites[metabolite] = float(coefficient)

    @property
    def reactants(self) -> Dict[Metabolite, float]:
        """Return consumed metabolites (negative stoichiometric coefficient)."""
        return {m: -c for m, c in self.metabolites.items() if c < 0}

    @property
    def products(self) -> Dict[Metabolite, float]:
        """Return produced metabolites (positive stoichiometric coefficient)."""
        return {m: c for m, c in self.metabolites.items() if c > 0}

    def build_reaction_string(self) -> str:
        """Format reaction as human-readable chemical equation."""

        def _format_side(met_dict: Dict[Metabolite, float]) -> str:
            parts: List[str] = []
            for met, coeff in met_dict.items():
                coeff_str = f"{coeff:g} " if coeff != 1.0 else ""
                parts.append(f"{coeff_str}{met.id}")
            return " + ".join(parts) if parts else "0"

        left = _format_side(self.reactants)
        right = _format_side(self.products)
        arrow = "<=>" if self.is_reversible else "-->"
        return f"{left} {arrow} {right}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reaction to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "metabolites": {m.id: coeff for m, coeff in self.metabolites.items()},
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "objective_coefficient": self.objective_coefficient,
            "gene_reaction_rule": self.gene_reaction_rule,
            "subsystem": self.subsystem,
        }


class MetabolicModel:
    """Genome-Scale Metabolic Network model."""

    def __init__(self, model_id: str, name: str = "") -> None:
        if not model_id:
            raise ValueError("Model ID cannot be empty.")
        self.id: str = model_id
        self.name: str = name
        self._metabolites: Dict[str, Metabolite] = {}
        self._reactions: Dict[str, Reaction] = {}
        self._genes: Dict[str, Gene] = {}

    @property
    def metabolites(self) -> List[Metabolite]:
        """Return list of all registered metabolites."""
        return list(self._metabolites.values())

    @property
    def reactions(self) -> List[Reaction]:
        """Return list of all registered reactions."""
        return list(self._reactions.values())

    @property
    def genes(self) -> List[Gene]:
        """Return list of all registered genes."""
        return list(self._genes.values())

    def add_metabolite(self, metabolite: Metabolite) -> None:
        """Register a metabolite in the model."""
        if metabolite.id in self._metabolites:
            raise ValueError(f"Metabolite '{metabolite.id}' already exists in model.")
        self._metabolites[metabolite.id] = metabolite

    def add_metabolites(self, metabolites: List[Metabolite]) -> None:
        """Register multiple metabolites."""
        for m in metabolites:
            self.add_metabolite(m)

    def get_metabolite(self, metabolite_id: str) -> Metabolite:
        """Retrieve a metabolite by ID."""
        if metabolite_id not in self._metabolites:
            raise KeyError(f"Metabolite '{metabolite_id}' not found in model.")
        return self._metabolites[metabolite_id]

    def add_gene(self, gene: Gene) -> None:
        """Register a gene in the model."""
        if gene.id not in self._genes:
            self._genes[gene.id] = gene

    def get_gene(self, gene_id: str) -> Gene:
        """Retrieve a gene by ID."""
        if gene_id not in self._genes:
            raise KeyError(f"Gene '{gene_id}' not found in model.")
        return self._genes[gene_id]

    def add_reaction(self, reaction: Reaction) -> None:
        """Register a reaction and auto-register referenced metabolites and genes."""
        if reaction.id in self._reactions:
            raise ValueError(f"Reaction '{reaction.id}' already exists in model.")

        for met in reaction.metabolites.keys():
            if met.id not in self._metabolites:
                self._metabolites[met.id] = met

        self._reactions[reaction.id] = reaction

    def add_reactions(self, reactions: List[Reaction]) -> None:
        """Register multiple reactions."""
        for rxn in reactions:
            self.add_reaction(rxn)

    def get_reaction(self, reaction_id: str) -> Reaction:
        """Retrieve a reaction by ID."""
        if reaction_id not in self._reactions:
            raise KeyError(f"Reaction '{reaction_id}' not found in model.")
        return self._reactions[reaction_id]

    def remove_reaction(self, reaction_id: str) -> None:
        """Remove a reaction from the model."""
        if reaction_id in self._reactions:
            del self._reactions[reaction_id]

    def set_objective(self, reaction_id: str, coefficient: float = 1.0) -> None:
        """Set a single reaction as the primary objective."""
        for rxn in self._reactions.values():
            rxn.objective_coefficient = coefficient if rxn.id == reaction_id else 0.0

    def stoichiometric_matrix(self) -> np.ndarray:
        """Construct the dense stoichiometric matrix S (m x n).

        Rows represent metabolites (m), columns represent reactions (n).
        """
        m = len(self._metabolites)
        n = len(self._reactions)
        s_matrix = np.zeros((m, n), dtype=np.float64)

        met_idx = {met.id: i for i, met in enumerate(self._metabolites.values())}

        for j, rxn in enumerate(self._reactions.values()):
            for met, coeff in rxn.metabolites.items():
                i = met_idx[met.id]
                s_matrix[i, j] = coeff

        return s_matrix

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bound vectors (v_min, v_max) as float64 numpy arrays."""
        rxns = list(self._reactions.values())
        lb = np.array([r.lower_bound for r in rxns], dtype=np.float64)
        ub = np.array([r.upper_bound for r in rxns], dtype=np.float64)
        return lb, ub

    def get_objective_vector(self) -> np.ndarray:
        """Return objective coefficient vector c as float64 numpy array."""
        return np.array(
            [r.objective_coefficient for r in self._reactions.values()],
            dtype=np.float64,
        )

    def extract_all_genes(self) -> Set[str]:
        """Extract all unique gene IDs declared in reaction GPR strings."""
        from flux_weave.gpr import extract_genes_from_rule

        genes: Set[str] = set()
        for rxn in self._reactions.values():
            if rxn.gene_reaction_rule:
                genes.update(extract_genes_from_rule(rxn.gene_reaction_rule))
        return genes

    def copy(self) -> MetabolicModel:
        """Create a deep copy of the metabolic model."""
        new_model = MetabolicModel(self.id, self.name)
        for m in self._metabolites.values():
            new_model.add_metabolite(
                Metabolite(
                    id=m.id,
                    name=m.name,
                    compartment=m.compartment,
                    formula=m.formula,
                    charge=m.charge,
                )
            )
        for g in self._genes.values():
            new_model.add_gene(Gene(id=g.id, name=g.name))

        for r in self._reactions.values():
            new_met_dict: Dict[Metabolite, float] = {}
            for met, coeff in r.metabolites.items():
                copied_met = new_model.get_metabolite(met.id)
                new_met_dict[copied_met] = coeff
            new_rxn = Reaction(
                id=r.id,
                name=r.name,
                metabolites=new_met_dict,
                lower_bound=r.lower_bound,
                upper_bound=r.upper_bound,
                objective_coefficient=r.objective_coefficient,
                gene_reaction_rule=r.gene_reaction_rule,
                subsystem=r.subsystem,
            )
            new_model.add_reaction(new_rxn)

        return new_model

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "metabolites": [
                {
                    "id": m.id,
                    "name": m.name,
                    "compartment": m.compartment,
                    "formula": m.formula,
                    "charge": m.charge,
                }
                for m in self._metabolites.values()
            ],
            "reactions": [r.to_dict() for r in self._reactions.values()],
            "genes": [{"id": g.id, "name": g.name} for g in self._genes.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetabolicModel:
        """Instantiate model from dictionary format."""
        model = cls(data["id"], data.get("name", ""))
        for m_data in data.get("metabolites", []):
            model.add_metabolite(
                Metabolite(
                    id=m_data["id"],
                    name=m_data.get("name", ""),
                    compartment=m_data.get("compartment", "c"),
                    formula=m_data.get("formula"),
                    charge=m_data.get("charge"),
                )
            )
        for g_data in data.get("genes", []):
            model.add_gene(Gene(id=g_data["id"], name=g_data.get("name", "")))

        for r_data in data.get("reactions", []):
            rxn_mets: Dict[Metabolite, float] = {}
            for met_id, coeff in r_data.get("metabolites", {}).items():
                rxn_mets[model.get_metabolite(met_id)] = float(coeff)
            rxn = Reaction(
                id=r_data["id"],
                name=r_data.get("name", ""),
                metabolites=rxn_mets,
                lower_bound=float(r_data.get("lower_bound", 0.0)),
                upper_bound=float(r_data.get("upper_bound", 1000.0)),
                objective_coefficient=float(r_data.get("objective_coefficient", 0.0)),
                gene_reaction_rule=r_data.get("gene_reaction_rule", ""),
                subsystem=r_data.get("subsystem", ""),
            )
            model.add_reaction(rxn)

        return model

    def to_json(self, indent: int = 2) -> str:
        """Export model as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> MetabolicModel:
        """Load model from JSON string."""
        return cls.from_dict(json.loads(json_str))
