"""
platform/lessons.py
-------------------
Definitions for all 7 STEM lessons on the Extending STEM across ASL platform.

Each lesson is a structured Python dict with the following keys:
    id              : Unique lesson identifier (matches search engine concept ID)
    name            : Display name of the lesson
    domain          : STEM domain (Physics, Chemistry, Biology, Mathematics)
    description     : Full educational description of the concept
    key_terms       : List of dicts {term, asl_gesture_id, definition}
    asl_gesture_ids : List of ASL letter IDs used to sign this concept
    difficulty_level: "Beginner" | "Intermediate" | "Advanced"
    learning_objectives: List of strings
    quiz_questions  : List of dicts {question, options, answer, explanation}
    related_lessons : List of lesson IDs for navigation suggestions
"""

from __future__ import annotations
from typing import List, Dict, Any


# ── Helper ────────────────────────────────────────────────────────────────────

def _quiz(question: str, options: List[str],
          answer: str, explanation: str) -> Dict:
    return {
        "question":    question,
        "options":     options,
        "answer":      answer,
        "explanation": explanation,
    }


def _term(term: str, gesture: str, definition: str) -> Dict:
    return {"term": term, "asl_gesture_id": gesture, "definition": definition}


# ── Lessons ───────────────────────────────────────────────────────────────────

LESSONS: List[Dict[str, Any]] = [

    # ─────────────────────────────────────────────────────────────────────────
    # Physics: Gravity
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "PHY_G",
        "name":   "Gravity",
        "domain": "Physics",
        "description": (
            "Gravity is the invisible force that pulls objects with mass toward each "
            "other. On Earth, every object is pulled toward the planet's centre at an "
            "acceleration of 9.8 m/s². Isaac Newton formalised this as the Universal "
            "Law of Gravitation: F = G·m₁·m₂ / r², where G is the gravitational "
            "constant, m₁ and m₂ are the masses of two objects, and r is the distance "
            "between them. Gravity keeps planets in orbit around the Sun, holds the "
            "atmosphere in place, and determines the weight of objects."
        ),
        "key_terms": [
            _term("Gravitational Force",  "G", "The attractive force between any two masses."),
            _term("Free Fall",            "F", "Motion under gravity alone, with no other forces acting."),
            _term("Weight",               "W", "The force exerted on a mass by gravity; W = mg."),
            _term("Orbit",                "O", "A curved path around a body due to balanced gravity and inertia."),
            _term("Gravitational Field",  "G", "A region of space where a gravitational force acts on masses."),
        ],
        "asl_gesture_ids": ["G", "D", "F"],
        "difficulty_level": "Beginner",
        "learning_objectives": [
            "Define gravity as an attractive force between masses.",
            "State Newton's Universal Law of Gravitation and identify each variable.",
            "Calculate the weight of an object given its mass (W = mg).",
            "Explain why the Moon orbits the Earth using gravitational force.",
            "Distinguish between mass and weight.",
        ],
        "quiz_questions": [
            _quiz(
                "What is the approximate acceleration due to gravity on Earth's surface?",
                ["1.6 m/s²", "9.8 m/s²", "6.7 m/s²", "3.7 m/s²"],
                "9.8 m/s²",
                "Earth's gravitational acceleration is approximately 9.8 m/s². "
                "The Moon's is about 1.6 m/s² and Mars's is 3.7 m/s².",
            ),
            _quiz(
                "Which factor does NOT affect the gravitational force between two objects?",
                ["Mass of object 1", "Mass of object 2", "Distance between objects",
                 "Colour of the objects"],
                "Colour of the objects",
                "Newton's Law of Gravitation only depends on mass and distance — "
                "physical colour has no effect.",
            ),
            _quiz(
                "A 10 kg object is dropped from rest. What is its weight on Earth?",
                ["10 N", "98 N", "9.8 N", "100 N"],
                "98 N",
                "Weight = mass × g = 10 kg × 9.8 m/s² = 98 N.",
            ),
        ],
        "related_lessons": ["PHY_F", "MAT_A"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Physics: Force
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "PHY_F",
        "name":   "Force",
        "domain": "Physics",
        "description": (
            "A force is any interaction that, when unopposed, changes the motion of "
            "an object. Forces are vectors — they have both magnitude and direction. "
            "Newton's three laws of motion describe how forces affect objects: "
            "(1) An object remains at rest or in uniform motion unless acted upon by "
            "a net external force. (2) F = ma — force equals mass times acceleration. "
            "(3) Every action force has an equal and opposite reaction force. "
            "Common forces include gravity, friction, tension, normal force, and "
            "electromagnetic force."
        ),
        "key_terms": [
            _term("Net Force",       "F", "The vector sum of all forces acting on an object."),
            _term("Friction",        "F", "A resistive force opposing relative motion between surfaces."),
            _term("Tension",         "T", "A pulling force transmitted through a rope or cable."),
            _term("Normal Force",    "N", "The contact force perpendicular to a surface."),
            _term("Newton",          "N", "The SI unit of force (1 N = 1 kg·m/s²)."),
        ],
        "asl_gesture_ids": ["F", "O", "R"],
        "difficulty_level": "Beginner",
        "learning_objectives": [
            "Define force as a vector quantity with magnitude and direction.",
            "State and apply Newton's three laws of motion.",
            "Distinguish between balanced and unbalanced forces.",
            "Identify common forces (gravity, friction, tension, normal).",
            "Calculate net force and resulting acceleration using F = ma.",
        ],
        "quiz_questions": [
            _quiz(
                "A 5 kg box is pushed with a net force of 20 N. What is its acceleration?",
                ["4 m/s²", "100 m/s²", "0.25 m/s²", "25 m/s²"],
                "4 m/s²",
                "a = F/m = 20 N / 5 kg = 4 m/s².",
            ),
            _quiz(
                "Which of Newton's laws explains why a rocket accelerates upward?",
                ["First Law", "Second Law", "Third Law", "Law of Gravitation"],
                "Third Law",
                "The rocket expels gas downward (action); the gas exerts an equal "
                "and opposite force upward on the rocket (reaction) — Newton's Third Law.",
            ),
            _quiz(
                "When forces on an object are balanced, the object must be:",
                ["Accelerating", "At rest or moving at constant velocity",
                 "Decelerating", "In free fall"],
                "At rest or moving at constant velocity",
                "Balanced forces mean zero net force, so by Newton's First Law "
                "there is no change in motion.",
            ),
        ],
        "related_lessons": ["PHY_G", "MAT_A"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Chemistry: Atoms
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "CHE_A",
        "name":   "Atoms",
        "domain": "Chemistry",
        "description": (
            "Atoms are the smallest units of ordinary matter that retain the chemical "
            "properties of an element. Every atom consists of a dense nucleus containing "
            "positively charged protons and electrically neutral neutrons, surrounded by "
            "a cloud of negatively charged electrons. The atomic number (number of protons) "
            "uniquely identifies each element on the periodic table. Electrons occupy "
            "discrete energy levels (shells or orbitals), and chemical behaviour is largely "
            "determined by the number of electrons in the outermost shell (valence electrons)."
        ),
        "key_terms": [
            _term("Proton",          "P", "Positively charged particle in the nucleus; determines element identity."),
            _term("Neutron",         "N", "Neutral particle in the nucleus; affects atomic mass and stability."),
            _term("Electron",        "E", "Negatively charged particle orbiting the nucleus; drives chemistry."),
            _term("Atomic Number",   "A", "The number of protons in an atom's nucleus."),
            _term("Valence Electron","V", "Outer-shell electrons that participate in chemical bonding."),
        ],
        "asl_gesture_ids": ["A", "T", "M"],
        "difficulty_level": "Intermediate",
        "learning_objectives": [
            "Describe the structure of an atom (nucleus, electron cloud).",
            "Define atomic number and atomic mass.",
            "Explain the role of valence electrons in chemical bonding.",
            "Identify an element's position on the periodic table from its atomic number.",
            "Distinguish between isotopes of the same element.",
        ],
        "quiz_questions": [
            _quiz(
                "What determines which element an atom belongs to?",
                ["Number of neutrons", "Number of electrons", "Number of protons", "Atomic mass"],
                "Number of protons",
                "The atomic number (proton count) uniquely identifies each element.",
            ),
            _quiz(
                "Carbon-12 and Carbon-14 are examples of:",
                ["Ions", "Isotopes", "Isomers", "Allotropes"],
                "Isotopes",
                "Both have 6 protons (carbon), but different neutron counts — making them isotopes.",
            ),
            _quiz(
                "Which subatomic particle is responsible for chemical bonding?",
                ["Protons", "Neutrons", "Electrons", "Quarks"],
                "Electrons",
                "Valence electrons are shared or transferred during chemical reactions.",
            ),
        ],
        "related_lessons": ["CHE_M", "BIO_D"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Chemistry: Molecules
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "CHE_M",
        "name":   "Molecules",
        "domain": "Chemistry",
        "description": (
            "A molecule is formed when two or more atoms bond together through shared "
            "or transferred electrons. Covalent bonds involve sharing of electrons "
            "(e.g., H₂O, CO₂), while ionic bonds involve electron transfer (e.g., NaCl). "
            "The geometry of a molecule (its 3-D shape) is determined by the "
            "VSEPR theory (Valence Shell Electron Pair Repulsion) and directly affects "
            "its physical and chemical properties. Molecular formulas describe which atoms "
            "are present; structural formulas show how they are connected."
        ),
        "key_terms": [
            _term("Covalent Bond",   "C", "A bond formed by sharing electrons between atoms."),
            _term("Ionic Bond",      "I", "A bond formed by electrostatic attraction between oppositely charged ions."),
            _term("Molecular Formula","M","Notation showing the type and count of atoms in a molecule."),
            _term("VSEPR Theory",    "V", "Model predicting molecular geometry from electron pair repulsion."),
            _term("Polarity",        "P", "Uneven distribution of electron density, giving partial charges."),
        ],
        "asl_gesture_ids": ["M", "O", "L"],
        "difficulty_level": "Intermediate",
        "learning_objectives": [
            "Distinguish between covalent and ionic bonding.",
            "Interpret molecular formulas (e.g., H₂O has 2 H + 1 O).",
            "Predict molecular shape using VSEPR theory.",
            "Explain why water is a polar molecule.",
            "Describe hydrogen bonding and its effects on water properties.",
        ],
        "quiz_questions": [
            _quiz(
                "How many atoms are in one molecule of glucose (C₆H₁₂O₆)?",
                ["6", "12", "24", "18"],
                "24",
                "6 carbon + 12 hydrogen + 6 oxygen = 24 atoms total.",
            ),
            _quiz(
                "Which type of bond is found in water (H₂O)?",
                ["Ionic", "Metallic", "Covalent", "Van der Waals"],
                "Covalent",
                "Oxygen shares electrons with each hydrogen atom via covalent bonds.",
            ),
            _quiz(
                "What shape does a water molecule have?",
                ["Linear", "Trigonal planar", "Bent", "Tetrahedral"],
                "Bent",
                "Two bonding pairs + two lone pairs on oxygen create a bent geometry (~104.5°).",
            ),
        ],
        "related_lessons": ["CHE_A", "BIO_C"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Biology: Cells
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "BIO_C",
        "name":   "Cells",
        "domain": "Biology",
        "description": (
            "The cell is the fundamental structural and functional unit of all living "
            "organisms. Prokaryotic cells (bacteria, archaea) lack a membrane-bound nucleus; "
            "their genetic material floats freely in the cytoplasm. Eukaryotic cells "
            "(plants, animals, fungi, protists) have a nucleus and membrane-bound organelles. "
            "Key organelles include: the nucleus (DNA storage and transcription), "
            "mitochondria (ATP energy production via cellular respiration), ribosomes "
            "(protein synthesis), the endoplasmic reticulum (protein/lipid processing), "
            "and — in plants — chloroplasts (photosynthesis). Cells reproduce via mitosis "
            "(somatic cells) and meiosis (sex cells)."
        ),
        "key_terms": [
            _term("Cell Membrane",   "C", "Phospholipid bilayer controlling what enters and exits the cell."),
            _term("Nucleus",         "N", "Organelle housing DNA and directing cellular activities."),
            _term("Mitochondria",    "M", "Organelle that produces ATP through cellular respiration."),
            _term("Ribosome",        "R", "Molecular machine that synthesises proteins from mRNA."),
            _term("Chloroplast",     "C", "Plant organelle that converts sunlight to glucose (photosynthesis)."),
        ],
        "asl_gesture_ids": ["C", "E", "L"],
        "difficulty_level": "Intermediate",
        "learning_objectives": [
            "State the cell theory and its three principles.",
            "Compare prokaryotic and eukaryotic cells.",
            "Identify key organelles and their functions.",
            "Describe the stages of mitosis.",
            "Explain how cells obtain and use energy.",
        ],
        "quiz_questions": [
            _quiz(
                "Which organelle is known as the 'powerhouse of the cell'?",
                ["Nucleus", "Ribosome", "Mitochondria", "Chloroplast"],
                "Mitochondria",
                "Mitochondria produce ATP via cellular respiration, supplying energy for cell processes.",
            ),
            _quiz(
                "Which of the following is found in plant cells but NOT animal cells?",
                ["Mitochondria", "Cell wall", "Ribosome", "Cell membrane"],
                "Cell wall",
                "Plant cells have a rigid cell wall made of cellulose; animal cells do not.",
            ),
            _quiz(
                "How many daughter cells are produced by one cell undergoing mitosis?",
                ["1", "2", "4", "8"],
                "2",
                "Mitosis produces two genetically identical daughter cells from one parent cell.",
            ),
        ],
        "related_lessons": ["BIO_D", "CHE_M"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Biology: DNA
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "BIO_D",
        "name":   "DNA",
        "domain": "Biology",
        "description": (
            "DNA (Deoxyribonucleic Acid) is the molecule that encodes genetic information "
            "in all known living organisms and many viruses. Its structure — the double helix "
            "discovered by Watson, Crick, Franklin, and Wilkins in 1953 — consists of two "
            "polynucleotide strands wound around each other. Each nucleotide contains a sugar "
            "(deoxyribose), a phosphate group, and one of four nitrogenous bases: Adenine (A), "
            "Thymine (T), Guanine (G), and Cytosine (C). Base-pairing rules (A–T, G–C) hold "
            "the two strands together. The sequence of bases acts as a code: genes are "
            "transcribed into mRNA and then translated into proteins."
        ),
        "key_terms": [
            _term("Nucleotide",      "N", "The monomer of DNA: sugar + phosphate + nitrogenous base."),
            _term("Double Helix",    "D", "The spiral ladder structure of the DNA molecule."),
            _term("Base Pair",       "B", "Complementary bases held by hydrogen bonds (A–T, G–C)."),
            _term("Transcription",   "T", "Copying DNA into mRNA in the nucleus."),
            _term("Translation",     "T", "Decoding mRNA into a protein at the ribosome."),
        ],
        "asl_gesture_ids": ["D", "N", "A"],
        "difficulty_level": "Advanced",
        "learning_objectives": [
            "Describe the double-helix structure of DNA.",
            "Identify the four nitrogenous bases and their complementary pairing.",
            "Explain DNA replication and why it is semi-conservative.",
            "Outline the central dogma: DNA → RNA → Protein.",
            "Define mutation and give examples of types (substitution, insertion, deletion).",
        ],
        "quiz_questions": [
            _quiz(
                "Which base pairs with Adenine in DNA?",
                ["Guanine", "Cytosine", "Uracil", "Thymine"],
                "Thymine",
                "In DNA, Adenine (A) pairs with Thymine (T) via two hydrogen bonds.",
            ),
            _quiz(
                "Where does transcription take place in a eukaryotic cell?",
                ["Ribosome", "Cytoplasm", "Mitochondria", "Nucleus"],
                "Nucleus",
                "Transcription (DNA → mRNA) occurs in the nucleus, where the DNA is stored.",
            ),
            _quiz(
                "Which term describes a change in the DNA base sequence?",
                ["Transcription", "Mutation", "Translation", "Replication"],
                "Mutation",
                "A mutation is any heritable change in the nucleotide sequence of DNA.",
            ),
        ],
        "related_lessons": ["BIO_C", "CHE_A"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Mathematics: Algebra
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id":     "MAT_A",
        "name":   "Algebra",
        "domain": "Mathematics",
        "description": (
            "Algebra is the branch of mathematics that uses symbols (variables) to represent "
            "numbers and express mathematical relationships. Variables allow us to write "
            "general formulas and solve for unknown quantities. Key topics include: "
            "expressions (combinations of variables and constants), equations (statements "
            "of equality between two expressions), linear equations (degree 1, graph as a "
            "straight line), quadratic equations (degree 2, solved by factoring, completing "
            "the square, or the quadratic formula), and systems of equations (multiple "
            "equations solved simultaneously). Algebra is the foundation of all higher "
            "mathematics and is used extensively in physics, chemistry, and biology."
        ),
        "key_terms": [
            _term("Variable",        "V", "A symbol (e.g., x) representing an unknown or changing quantity."),
            _term("Equation",        "E", "A mathematical statement asserting the equality of two expressions."),
            _term("Coefficient",     "C", "The numerical factor multiplying a variable in a term."),
            _term("Linear Equation", "L", "An equation of degree 1; its graph is a straight line (y = mx + b)."),
            _term("Quadratic",       "Q", "An equation of degree 2; general form ax² + bx + c = 0."),
        ],
        "asl_gesture_ids": ["A", "L", "G"],
        "difficulty_level": "Beginner",
        "learning_objectives": [
            "Distinguish between expressions, equations, and inequalities.",
            "Solve single-variable linear equations.",
            "Interpret the slope and y-intercept of a linear equation.",
            "Solve quadratic equations using the quadratic formula.",
            "Set up and solve a system of two linear equations.",
        ],
        "quiz_questions": [
            _quiz(
                "Solve for x: 3x + 6 = 21",
                ["x = 5", "x = 9", "x = 7", "x = 3"],
                "x = 5",
                "3x = 21 − 6 = 15, so x = 15/3 = 5.",
            ),
            _quiz(
                "What is the slope of the line y = 4x − 7?",
                ["−7", "4", "7", "−4"],
                "4",
                "In y = mx + b, m is the slope. Here m = 4.",
            ),
            _quiz(
                "Which formula solves any quadratic equation ax² + bx + c = 0?",
                ["x = −b/a", "x = (−b ± √(b²−4ac)) / 2a",
                 "x = b/2a", "x = √(b²−4ac)"],
                "x = (−b ± √(b²−4ac)) / 2a",
                "The quadratic formula gives both roots of any quadratic equation.",
            ),
        ],
        "related_lessons": ["PHY_G", "PHY_F"],
    },
]


# ── Index ─────────────────────────────────────────────────────────────────────

LESSON_INDEX: dict = {lesson["id"]: lesson for lesson in LESSONS}


def get_lesson(lesson_id: str) -> dict | None:
    """Return a lesson dict by ID, or None if not found."""
    return LESSON_INDEX.get(lesson_id)


def get_lesson_by_name(name: str) -> dict | None:
    """Return a lesson dict by name (case-insensitive), or None."""
    name_lower = name.lower().strip()
    for lesson in LESSONS:
        if lesson["name"].lower() == name_lower:
            return lesson
    return None


def get_lessons_by_domain(domain: str) -> list:
    """Return all lessons belonging to a STEM domain (case-insensitive)."""
    return [l for l in LESSONS if l["domain"].lower() == domain.lower()]


def list_all_lessons() -> list:
    """Return all 7 lesson dicts."""
    return list(LESSONS)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEM Lessons – Self-test")
    print("=" * 60)

    print(f"\nTotal lessons loaded : {len(LESSONS)}")
    for lesson in LESSONS:
        print(f"  [{lesson['id']}]  {lesson['domain']:<12}  {lesson['name']:<12}  "
              f"({lesson['difficulty_level']})  "
              f"  {len(lesson['quiz_questions'])} quiz questions")

    print()
    lesson = get_lesson_by_name("DNA")
    if lesson:
        print(f"Lesson lookup by name 'DNA': {lesson['name']} | {lesson['id']}")
        print(f"  Key terms: {[t['term'] for t in lesson['key_terms']]}")
        print(f"  Quiz q1  : {lesson['quiz_questions'][0]['question']}")

    print()
    print("✅  Lessons module verified successfully.")
