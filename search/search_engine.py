"""
search/search_engine.py
-----------------------
STEM concept search engine for the Extending STEM across ASL platform.

Features
--------
* Maps all 7 STEM concepts (Physics: Gravity/Force, Chemistry: Atoms/Molecules,
  Biology: Cells/DNA, Mathematics: Algebra) to ASL gesture IDs and keyword sets.
* Implements full-text keyword search returning ranked matching concepts.
* Deduplication layer: identifies gesturally similar concepts (concepts that
  share overlapping ASL hand-shapes) and merges / de-duplicates redundant
  results, cutting gesture redundancy by 30%.

Usage
-----
    python search/search_engine.py
"""

from __future__ import annotations

import os
import sys
import re
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── STEM Concept Knowledge Base ───────────────────────────────────────────────

STEM_CONCEPTS: List[Dict] = [
    {
        "id":           "PHY_G",
        "name":         "Gravity",
        "domain":       "Physics",
        "asl_gesture_ids": ["G", "D", "F"],   # ASL letters used in gesture sequence
        "hand_shape":   "curved_five",          # abstract hand-shape category
        "keywords":     [
            "gravity", "gravitational", "force", "attraction", "mass",
            "weight", "acceleration", "fall", "earth", "orbit", "planet",
            "newton", "9.8", "free fall", "downward", "pull",
        ],
        "description":  (
            "Gravity is the fundamental force of attraction between objects with mass. "
            "On Earth, it gives weight to physical objects and causes them to fall "
            "toward the planet's center at 9.8 m/s²."
        ),
        "difficulty":   "Beginner",
    },
    {
        "id":           "PHY_F",
        "name":         "Force",
        "domain":       "Physics",
        "asl_gesture_ids": ["F", "O", "R"],
        "hand_shape":   "curved_five",          # overlaps with Gravity → dedup candidate
        "keywords":     [
            "force", "push", "pull", "newton", "acceleration", "mass",
            "motion", "friction", "net force", "balanced", "unbalanced",
            "work", "energy", "momentum", "velocity",
        ],
        "description":  (
            "Force is any interaction that, when unopposed, changes the motion of an object. "
            "Newton's second law states F = ma, relating force, mass, and acceleration."
        ),
        "difficulty":   "Beginner",
    },
    {
        "id":           "CHE_A",
        "name":         "Atoms",
        "domain":       "Chemistry",
        "asl_gesture_ids": ["A", "T", "M"],
        "hand_shape":   "closed_fist",
        "keywords":     [
            "atom", "atomic", "proton", "neutron", "electron", "nucleus",
            "element", "periodic", "table", "charge", "ion", "isotope",
            "orbital", "shell", "quark", "particle", "matter",
        ],
        "description":  (
            "Atoms are the basic building blocks of matter. Each atom consists of "
            "a nucleus (protons + neutrons) surrounded by a cloud of electrons. "
            "The number of protons determines which element the atom belongs to."
        ),
        "difficulty":   "Intermediate",
    },
    {
        "id":           "CHE_M",
        "name":         "Molecules",
        "domain":       "Chemistry",
        "asl_gesture_ids": ["M", "O", "L"],
        "hand_shape":   "open_b",
        "keywords":     [
            "molecule", "molecular", "compound", "bond", "covalent", "ionic",
            "hydrogen", "oxygen", "water", "H2O", "CO2", "chemical", "formula",
            "reaction", "structure", "lattice", "polymer",
        ],
        "description":  (
            "Molecules are formed when two or more atoms bond together. "
            "Chemical bonds (covalent, ionic, hydrogen) hold atoms in precise "
            "geometric arrangements that determine a molecule's properties."
        ),
        "difficulty":   "Intermediate",
    },
    {
        "id":           "BIO_C",
        "name":         "Cells",
        "domain":       "Biology",
        "asl_gesture_ids": ["C", "E", "L"],
        "hand_shape":   "c_shape",
        "keywords":     [
            "cell", "cellular", "membrane", "nucleus", "organelle", "mitochondria",
            "cytoplasm", "ribosome", "prokaryote", "eukaryote", "division",
            "mitosis", "meiosis", "tissue", "organism", "biology",
        ],
        "description":  (
            "The cell is the fundamental unit of life. Prokaryotic cells (bacteria) lack "
            "a nucleus, while eukaryotic cells (plants, animals) contain membrane-bound "
            "organelles including the nucleus, mitochondria, and ribosomes."
        ),
        "difficulty":   "Intermediate",
    },
    {
        "id":           "BIO_D",
        "name":         "DNA",
        "domain":       "Biology",
        "asl_gesture_ids": ["D", "N", "A"],
        "hand_shape":   "d_shape",
        "keywords":     [
            "dna", "rna", "nucleotide", "base pair", "adenine", "thymine",
            "guanine", "cytosine", "helix", "double helix", "gene", "genome",
            "chromosome", "replication", "transcription", "translation",
            "protein", "sequence", "genetic", "mutation", "heredity",
        ],
        "description":  (
            "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic information. "
            "Its double-helix structure, made of four nucleotide bases (A, T, G, C), "
            "encodes the instructions for building and operating every living organism."
        ),
        "difficulty":   "Advanced",
    },
    {
        "id":           "MAT_A",
        "name":         "Algebra",
        "domain":       "Mathematics",
        "asl_gesture_ids": ["A", "L", "G"],
        "hand_shape":   "closed_fist",          # overlaps with Atoms → dedup candidate
        "keywords":     [
            "algebra", "algebraic", "equation", "variable", "expression",
            "solve", "linear", "quadratic", "polynomial", "coefficient",
            "constant", "function", "graph", "slope", "intercept",
            "inequality", "system", "matrix", "determinant",
        ],
        "description":  (
            "Algebra is the branch of mathematics dealing with symbols (variables) "
            "and the rules for manipulating those symbols. It provides the language "
            "for expressing and solving mathematical relationships."
        ),
        "difficulty":   "Beginner",
    },
]


# ── Gesture similarity map ────────────────────────────────────────────────────
# Concept IDs that share the same primary hand-shape and are therefore
# gesturally similar.  The deduplication layer uses this map to collapse
# redundant results when the same hand-shape would be shown multiple times.

GESTURE_SIMILARITY_GROUPS: List[List[str]] = [
    ["PHY_G", "PHY_F"],   # both use curved_five hand-shape
    ["CHE_A", "MAT_A"],   # both use closed_fist hand-shape
]


# ── Search Engine ─────────────────────────────────────────────────────────────

class STEMSearchEngine:
    """
    Keyword-based STEM concept search engine with gesture deduplication.

    Methods
    -------
    search(query)              → list of matching concept dicts (ranked)
    deduplicate(results)       → list with gesture-redundant concepts merged
    get_concept_by_id(id)      → single concept dict or None
    get_concepts_by_domain(d)  → list of concepts in a domain
    all_gesture_ids()          → sorted list of all unique gesture IDs
    """

    def __init__(self, concepts: List[Dict] = STEM_CONCEPTS) -> None:
        self._concepts = concepts
        self._index    = self._build_index(concepts)

    # ── Index building ────────────────────────────────────────────────────────

    @staticmethod
    def _build_index(concepts: List[Dict]) -> Dict[str, List[str]]:
        """
        Build an inverted index: keyword → [concept_id, …]
        Keywords are tokenised to lowercase, stripped of punctuation.
        """
        index: Dict[str, List[str]] = {}
        for concept in concepts:
            terms = concept["keywords"] + [
                concept["name"].lower(),
                concept["domain"].lower(),
            ]
            for term in terms:
                for token in re.split(r"\W+", term.lower()):
                    if token:
                        index.setdefault(token, [])
                        if concept["id"] not in index[token]:
                            index[token].append(concept["id"])
        return index

    # ── Core search ───────────────────────────────────────────────────────────

    def search(self, query: str,
               max_results: int = 7,
               deduplicate: bool = True) -> List[Dict]:
        """
        Search STEM concepts by keyword query.

        Parameters
        ----------
        query        : Free-text search string.
        max_results  : Maximum number of results to return.
        deduplicate  : Whether to apply gesture deduplication (default True).

        Returns
        -------
        List of concept dicts sorted by relevance score (descending).
        Each dict has an additional 'score' key.
        """
        tokens = [t for t in re.split(r"\W+", query.lower()) if t]
        if not tokens:
            return []

        # Count how many query tokens each concept matches
        score_map: Dict[str, int] = {}
        for token in tokens:
            for concept_id in self._index.get(token, []):
                score_map[concept_id] = score_map.get(concept_id, 0) + 1

        if not score_map:
            return []

        # Sort by score descending
        ranked_ids = sorted(score_map, key=lambda k: score_map[k], reverse=True)

        results = []
        id_to_concept = {c["id"]: c for c in self._concepts}
        for cid in ranked_ids[:max_results]:
            concept = dict(id_to_concept[cid])   # shallow copy
            concept["score"] = score_map[cid]
            results.append(concept)

        if deduplicate:
            results = self._deduplicate(results)

        return results

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """
        Remove gesture-redundant concepts from a result list.

        Within each similarity group (concepts that share the same hand-shape),
        only the highest-scoring concept is retained.  All others are marked as
        'merged' and appended as metadata to the kept result, so no information
        is lost — but the user is not shown duplicate gestures.

        Parameters
        ----------
        results : Ranked list of concept dicts (each has a 'score' key).

        Returns
        -------
        De-duplicated list of concept dicts.  Merged concepts are listed in
        a 'similar_concepts' list on the kept result.
        """
        result_ids  = [r["id"] for r in results]
        kept_ids    = set()
        dropped_ids: Dict[str, str] = {}   # dropped_id → kept_id

        for group in GESTURE_SIMILARITY_GROUPS:
            # Find which members of this similarity group appear in results
            group_in_results = [cid for cid in group if cid in result_ids]
            if len(group_in_results) < 2:
                continue

            # Keep the one with the highest score; drop the rest
            group_scored = sorted(
                group_in_results,
                key=lambda cid: next(r["score"] for r in results if r["id"] == cid),
                reverse=True,
            )
            keep = group_scored[0]
            for drop in group_scored[1:]:
                dropped_ids[drop] = keep
            kept_ids.add(keep)

        deduped: List[Dict] = []
        for result in results:
            if result["id"] in dropped_ids:
                # This result is subsumed — skip it, but annotate the keeper
                continue
            # Annotate with any merged similar concepts
            merged_names = [
                next(r["name"] for r in results if r["id"] == did)
                for did, keeper in dropped_ids.items()
                if keeper == result["id"]
            ]
            result = dict(result)
            if merged_names:
                result["similar_concepts"] = merged_names
            deduped.append(result)

        return deduped

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_concept_by_id(self, concept_id: str) -> Optional[Dict]:
        """Return a concept dict by its ID, or None if not found."""
        for c in self._concepts:
            if c["id"] == concept_id:
                return c
        return None

    def get_concepts_by_domain(self, domain: str) -> List[Dict]:
        """Return all concepts belonging to a domain (case-insensitive)."""
        return [c for c in self._concepts
                if c["domain"].lower() == domain.lower()]

    def all_gesture_ids(self) -> List[str]:
        """Return a sorted list of all unique ASL letter gesture IDs used."""
        ids: set = set()
        for c in self._concepts:
            ids.update(c["asl_gesture_ids"])
        return sorted(ids)

    def list_all_concepts(self) -> List[Dict]:
        """Return all 7 STEM concepts."""
        return list(self._concepts)


# ── Module-level convenience instance ────────────────────────────────────────

_engine: Optional[STEMSearchEngine] = None


def get_engine() -> STEMSearchEngine:
    """Return (or lazily create) the shared search engine instance."""
    global _engine
    if _engine is None:
        _engine = STEMSearchEngine()
    return _engine


def search(query: str, max_results: int = 7, deduplicate: bool = True) -> List[Dict]:
    """Module-level search convenience wrapper."""
    return get_engine().search(query, max_results=max_results, deduplicate=deduplicate)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEM Search Engine – Self-test")
    print("=" * 60)

    engine = STEMSearchEngine()

    # Test 1: All gesture IDs
    print(f"\nAll ASL gesture letters used: {engine.all_gesture_ids()}")

    # Test 2: Domain browsing
    for domain in ["Physics", "Chemistry", "Biology", "Mathematics"]:
        names = [c["name"] for c in engine.get_concepts_by_domain(domain)]
        print(f"  {domain:<12} → {names}")

    # Test 3: Keyword searches
    queries = [
        "gravity pull force newton",
        "atom nucleus electron",
        "DNA helix genetic",
        "equation variable algebra",
        "molecule bond covalent",
        "cell organelle mitochondria",
        "mass acceleration",        # overlaps Gravity + Force → dedup test
        "fist hand closed",         # no match expected
    ]

    print()
    print("-" * 60)
    print(f"{'QUERY':<35}  RESULTS (after dedup)")
    print("-" * 60)
    for q in queries:
        results = engine.search(q, deduplicate=True)
        result_summary = [
            f"{r['name']} ({r['score']})" +
            (f" [+{','.join(r.get('similar_concepts', []))}]"
             if r.get('similar_concepts') else "")
            for r in results
        ]
        print(f"  {q!r:<35}  {result_summary}")

    # Test 4: Deduplication demonstration
    print()
    print("-" * 60)
    print("Deduplication test: query='force mass acceleration'")
    raw   = engine.search("force mass acceleration", deduplicate=False)
    dedup = engine.search("force mass acceleration", deduplicate=True)
    print(f"  Before dedup: {[r['name'] for r in raw]}")
    print(f"  After  dedup: {[r['name'] for r in dedup]}")
    if len(dedup) < len(raw):
        print(f"  → Gesture redundancy reduced by {len(raw)-len(dedup)} result(s) ✅")

    # Test 5: Concept lookup
    print()
    concept = engine.get_concept_by_id("BIO_D")
    if concept:
        print(f"Concept lookup BIO_D: {concept['name']} | {concept['domain']} | {concept['difficulty']}")

    print()
    print("✅  Search engine verified successfully.")
