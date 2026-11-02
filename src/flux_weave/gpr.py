"""Gene-Protein-Reaction (GPR) Boolean association parser and AST evaluator.

Handles arbitrary nested Boolean logic:
- 'and' expresses multi-subunit / heteromeric enzyme complexes (all required)
- 'or' expresses isoenzymes / alternative pathways (at least one sufficient)
- Nested parentheses and gene symbols (e.g., '(b0116 and b0720) or b3916')
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Union


class BooleanNode(ABC):
    """Abstract syntax tree node for GPR Boolean logic."""

    @abstractmethod
    def evaluate(self, gene_states: Dict[str, bool]) -> bool:
        """Evaluate node given boolean state of each gene."""
        pass

    @abstractmethod
    def get_genes(self) -> Set[str]:
        """Return set of all gene identifiers referenced in subtree."""
        pass


class GeneNode(BooleanNode):
    """Leaf node representing an individual gene."""

    def __init__(self, gene_id: str) -> None:
        self.gene_id = gene_id.strip()

    def evaluate(self, gene_states: Dict[str, bool]) -> bool:
        return gene_states.get(self.gene_id, True)

    def get_genes(self) -> Set[str]:
        return {self.gene_id}

    def __repr__(self) -> str:
        return self.gene_id


class AndNode(BooleanNode):
    """Conjunction node representing an enzyme complex (all subunits required)."""

    def __init__(self, children: List[BooleanNode]) -> None:
        self.children = children

    def evaluate(self, gene_states: Dict[str, bool]) -> bool:
        return all(c.evaluate(gene_states) for c in self.children)

    def get_genes(self) -> Set[str]:
        res: Set[str] = set()
        for c in self.children:
            res.update(c.get_genes())
        return res

    def __repr__(self) -> str:
        return f"({' and '.join(repr(c) for c in self.children)})"


class OrNode(BooleanNode):
    """Disjunction node representing isoenzymes (any catalyst sufficient)."""

    def __init__(self, children: List[BooleanNode]) -> None:
        self.children = children

    def evaluate(self, gene_states: Dict[str, bool]) -> bool:
        return any(c.evaluate(gene_states) for c in self.children)

    def get_genes(self) -> Set[str]:
        res: Set[str] = set()
        for c in self.children:
            res.update(c.get_genes())
        return res

    def __repr__(self) -> str:
        return f"({' or '.join(repr(c) for c in self.children)})"


class GPRParser:
    """Recursive descent parser for GPR Boolean proposition strings."""

    def __init__(self, expression: str) -> None:
        self.tokens = self._tokenize(expression)
        self.pos = 0

    @staticmethod
    def _tokenize(expr: str) -> List[str]:
        # Token pattern matching parentheses, logical operators, and gene identifiers
        pattern = r"\(|\)|\band\b|\bor\b|[\w\.\:\-\+]+"
        raw_tokens = re.findall(pattern, expr, flags=re.IGNORECASE)
        # Normalize and/or to lowercase
        tokens: List[str] = []
        for t in raw_tokens:
            t_low = t.lower()
            tokens.append(t_low if t_low in ("and", "or") else t)
        return tokens

    def parse(self) -> BooleanNode:
        if not self.tokens:
            return OrNode([])  # Vacuous truth or empty
        res = self._parse_or()
        return res

    def _parse_or(self) -> BooleanNode:
        nodes = [self._parse_and()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "or":
            self.pos += 1
            nodes.append(self._parse_and())
        if len(nodes) == 1:
            return nodes[0]
        return OrNode(nodes)

    def _parse_and(self) -> BooleanNode:
        nodes = [self._parse_factor()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "and":
            self.pos += 1
            nodes.append(self._parse_factor())
        if len(nodes) == 1:
            return nodes[0]
        return AndNode(nodes)

    def _parse_factor(self) -> BooleanNode:
        if self.pos >= len(self.tokens):
            return GeneNode("")
        tok = self.tokens[self.pos]
        if tok == "(":
            self.pos += 1
            node = self._parse_or()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ")":
                self.pos += 1
            return node
        self.pos += 1
        return GeneNode(tok)


def parse_gpr(rule: str) -> BooleanNode:
    """Parse GPR string into executable AST BooleanNode."""
    clean_rule = rule.strip()
    if not clean_rule:
        return OrNode([])
    return GPRParser(clean_rule).parse()


def extract_genes_from_rule(rule: str) -> Set[str]:
    """Extract all unique gene symbols referenced in a GPR rule string."""
    clean = rule.strip()
    if not clean:
        return set()
    node = parse_gpr(clean)
    return {g for g in node.get_genes() if g}


def eval_gpr(
    rule: str,
    active_or_knockout: Union[Set[str], Dict[str, bool]],
    is_knockout_set: bool = True,
) -> bool:
    """Evaluate whether an enzymatic reaction remains active.

    Parameters:
        rule: GPR rule string (e.g. '(g1 and g2) or g3')
        active_or_knockout: Set of knocked-out genes (if is_knockout_set=True)
                            or Dict mapping gene_id to boolean active state.
        is_knockout_set: If True, genes in active_or_knockout set are knocked out (False).
    """
    clean_rule = rule.strip()
    if not clean_rule:
        # Spontaneous / non-enzymatic reaction is always active
        return True

    node = parse_gpr(clean_rule)
    all_genes = node.get_genes()

    if is_knockout_set:
        knocked_out = (
            active_or_knockout
            if isinstance(active_or_knockout, set)
            else set(active_or_knockout)
        )
        gene_states = {g: (g not in knocked_out) for g in all_genes}
    else:
        gene_states = active_or_knockout if isinstance(active_or_knockout, dict) else {}

    return node.evaluate(gene_states)
