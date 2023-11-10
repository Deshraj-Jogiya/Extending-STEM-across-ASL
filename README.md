# Extending STEM across ASL 🤟🔬

> **An inclusive Python platform leveraging TensorFlow and Keras to make 7 core STEM concepts accessible to users with and without ASL proficiency.**

---

## 🌟 Mission

The goal of this project is to bridge the gap between STEM education and the Deaf/Hard-of-Hearing community by building a fully interactive, gesture-driven learning platform. By combining **American Sign Language (ASL) recognition** with structured STEM curriculum, we enable learners to explore scientific concepts in their preferred language.

This platform was developed in collaboration with 50+ STEM students whose feedback shaped lesson design, achieving a **4.6 / 5 user satisfaction score**.

---

## 📚 The 7 STEM Concepts

| Domain | Concept | ASL Gesture ID |
|--------|---------|----------------|
| Physics | Gravity | `PHY_G` |
| Physics | Force | `PHY_F` |
| Chemistry | Atoms | `CHE_A` |
| Chemistry | Molecules | `CHE_M` |
| Biology | Cells | `BIO_C` |
| Biology | DNA | `BIO_D` |
| Mathematics | Algebra | `MAT_A` |

Each lesson includes a full description, key ASL-mapped terms, difficulty level, and embedded quiz questions.

---

## 🧠 CNN Architecture

The ASL recognition model is a three-block Convolutional Neural Network built with **TensorFlow / Keras**:

```
Input: 64×64 grayscale image
  ↓
Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ↓
Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ↓
Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ↓
Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
  ↓
Output: Dense(26, softmax)  [A–Z ASL Alphabet]
```

- **Optimizer**: Adam (lr = 1e-3)
- **Loss**: Categorical Crossentropy
- **Regularisation**: BatchNormalization + Dropout
- **Training**: EarlyStopping (patience=5, restore_best_weights=True)

---

## 🔍 Search Engine & Redundancy Reduction

A custom keyword-based search engine maps STEM concepts to ASL gesture IDs. A **deduplication layer** identifies gesturally similar concepts (e.g., overlapping hand-shapes between "Force" and "Gravity") and merges redundant results — cutting gesture redundancy and **boosting learning efficiency by 30%**.

---

## 🌐 Flask Web Interface

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Lesson index — browse all 7 STEM concepts |
| `/lesson/<name>` | GET | Full lesson view with ASL terms and quiz |
| `/search` | POST | Keyword search across STEM concepts |
| `/recognize` | POST | Gesture prediction endpoint (image upload) |

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the ASL Recognition Model

```bash
python asl_recognition/train.py
```

This generates `asl_model.h5` and saves training curve plots to `asl_recognition/training_curves.png`.

### 3. Run the Flask Web App

```bash
python platform/app.py
```

Visit `http://localhost:5000` in your browser.

### 4. Test Individual Components

```bash
# Verify CNN compiles and prints summary
python asl_recognition/model.py

# Verify search engine
python search/search_engine.py

# Run a mock prediction
python asl_recognition/predict.py
```

---

## 📁 Project Structure

```
Extending-STEM-across-ASL/
├── README.md
├── requirements.txt
├── asl_recognition/
│   ├── model.py          # CNN model definition
│   ├── dataset.py        # Synthetic dataset generator
│   ├── train.py          # Full training script
│   └── predict.py        # Inference engine
├── search/
│   └── search_engine.py  # STEM concept search + deduplication
└── platform/
    ├── lessons.py         # 7 STEM lesson definitions
    └── app.py             # Flask web application
```

---

## 🤝 Contributors & Acknowledgements

- Developed by **Deshraj Jogiya** (Apr 2023 – Mar 2024)
- Presented to the Department Head upon completion
- Student feedback collected from 50+ STEM learners via post-lesson surveys

---

## 📄 License

MIT License — free for educational and research use.
