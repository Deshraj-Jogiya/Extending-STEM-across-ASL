"""
platform/app.py
---------------
Flask web application for the Extending STEM across ASL platform.

Routes
------
GET  /                         Lesson index — browse all 7 STEM concepts
GET  /lesson/<name>            Full lesson view with ASL terms and quiz
POST /search                   Keyword search across STEM concepts
POST /recognize                Gesture prediction endpoint (accepts image upload)

Run
---
    python platform/app.py

Then open  http://localhost:5000  in your browser.
"""

from __future__ import annotations

import os
import sys
import json
import base64
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import (
    Flask, request, jsonify, render_template_string,
    redirect, url_for,
)

from platform.lessons      import list_all_lessons, get_lesson_by_name, LESSONS
from search.search_engine  import STEMSearchEngine

app    = Flask(__name__)
engine = STEMSearchEngine()


# ══════════════════════════════════════════════════════════════════════════════
# HTML Templates (inline — no separate template files needed)
# ══════════════════════════════════════════════════════════════════════════════

BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --accent:   #58a6ff;
    --accent2:  #3fb950;
    --text:     #c9d1d9;
    --muted:    #8b949e;
    --danger:   #f85149;
    --radius:   12px;
  }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
  }

  nav {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    height: 56px;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  nav .brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    text-decoration: none;
    margin-right: auto;
  }

  nav form {
    display: flex;
    gap: 0.5rem;
  }

  nav input[type=text] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 0.35rem 0.75rem;
    font-size: 0.85rem;
    width: 200px;
    outline: none;
    transition: border-color .2s;
  }

  nav input[type=text]:focus { border-color: var(--accent); }

  nav button {
    background: var(--accent);
    color: #0d1117;
    border: none;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-weight: 600;
    cursor: pointer;
    font-size: 0.85rem;
    transition: opacity .2s;
  }

  nav button:hover { opacity: 0.85; }

  main { max-width: 1100px; margin: 2.5rem auto; padding: 0 1.5rem; }

  h1 { font-size: 2rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.5rem; }
  h2 { font-size: 1.3rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.75rem; }
  h3 { font-size: 1rem; font-weight: 600; color: var(--accent); margin-bottom: 0.5rem; }
  p  { color: var(--text); margin-bottom: 1rem; }

  .subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin-bottom: 2rem;
  }

  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.25rem;
    margin-top: 1.5rem;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    transition: border-color .25s, transform .25s;
    text-decoration: none;
    color: inherit;
    display: block;
  }

  .card:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
  }

  .card .domain-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
  }

  .badge-Physics     { background: #1a3a5c; color: #58a6ff; }
  .badge-Chemistry   { background: #1a3c2b; color: #3fb950; }
  .badge-Biology     { background: #3c1a2b; color: #f78166; }
  .badge-Mathematics { background: #2c1a3c; color: #bc8cff; }

  .card .lesson-title { font-size: 1.15rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.4rem; }
  .card .lesson-desc  { font-size: 0.85rem; color: var(--muted); line-height: 1.5; }
  .card .difficulty   { font-size: 0.75rem; color: var(--muted); margin-top: 0.75rem; }

  .section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .key-terms-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.75rem;
    margin-top: 0.75rem;
  }

  .term-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.85rem;
  }

  .term-card .gesture {
    display: inline-block;
    background: var(--accent);
    color: #0d1117;
    border-radius: 6px;
    font-weight: 700;
    font-size: 1.1rem;
    width: 36px;
    height: 36px;
    text-align: center;
    line-height: 36px;
    margin-bottom: 0.4rem;
  }

  .term-card .term-name { font-weight: 600; font-size: 0.9rem; color: #e6edf3; }
  .term-card .term-def  { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }

  .quiz-block { margin-top: 0.75rem; }

  .quiz-question {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
  }

  .quiz-question p { font-weight: 600; margin-bottom: 0.5rem; color: #e6edf3; }

  .quiz-options { list-style: none; }
  .quiz-options li {
    padding: 0.4rem 0.75rem;
    margin-bottom: 0.3rem;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: default;
    border: 1px solid transparent;
    transition: background .2s;
  }

  .quiz-options li:hover { background: #1c2128; }
  .quiz-options li.correct { border-color: var(--accent2); color: var(--accent2); }

  .objectives-list {
    padding-left: 1.2rem;
    margin-top: 0.5rem;
  }

  .objectives-list li { margin-bottom: 0.4rem; font-size: 0.9rem; color: var(--text); }

  .search-results { margin-top: 1.5rem; }
  .search-result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    transition: border-color .2s;
  }
  .search-result-card:hover { border-color: var(--accent); }
  .search-result-card a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .search-result-card a:hover { text-decoration: underline; }
  .score-pill {
    background: var(--accent);
    color: #0d1117;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    white-space: nowrap;
  }
  .similar-badge {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }

  .no-results { color: var(--muted); font-style: italic; margin-top: 1rem; }

  .back-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--accent);
    text-decoration: none;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    transition: opacity .2s;
  }
  .back-link:hover { opacity: 0.8; }

  .asl-gestures {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
  }

  .asl-letter {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 8px;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #fff;
    font-weight: 700;
    font-size: 1.2rem;
    box-shadow: 0 2px 8px rgba(31,111,235,.3);
  }

  footer {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
  }
</style>
"""


def _navbar(query: str = "") -> str:
    return f"""
    <nav>
      <a class="brand" href="/" id="nav-brand">🤟 STEM × ASL</a>
      <form method="POST" action="/search" id="nav-search-form">
        <input type="text" name="query" id="nav-search-input"
               placeholder="Search STEM concepts…" value="{query}" autocomplete="off">
        <button type="submit" id="nav-search-btn">Search</button>
      </form>
    </nav>
    """


def _domain_badge(domain: str) -> str:
    return f'<span class="domain-badge badge-{domain}">{domain}</span>'


def _difficulty_color(level: str) -> str:
    colors = {"Beginner": "#3fb950", "Intermediate": "#f0883e", "Advanced": "#f85149"}
    return colors.get(level, "#8b949e")


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Lesson index — browse all 7 STEM concepts."""
    lessons = list_all_lessons()

    cards_html = ""
    for lesson in lessons:
        url  = url_for("lesson_view", name=lesson["name"].lower())
        desc = lesson["description"][:140] + "…"
        cards_html += f"""
        <a class="card" href="{url}" id="card-{lesson['id']}">
          {_domain_badge(lesson['domain'])}
          <div class="lesson-title">{lesson['name']}</div>
          <div class="lesson-desc">{desc}</div>
          <div class="difficulty" style="color:{_difficulty_color(lesson['difficulty_level'])}">
            ⬤ {lesson['difficulty_level']}
          </div>
        </a>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Extending STEM across ASL | Home</title>
  <meta name="description"
    content="An inclusive platform covering 7 STEM concepts with ASL gesture recognition.">
  {BASE_CSS}
</head>
<body>
  {_navbar()}
  <main>
    <h1>Extending STEM across ASL 🤟</h1>
    <p class="subtitle">
      An inclusive education platform — explore 7 core STEM concepts with integrated
      American Sign Language gesture recognition.
    </p>
    <h2>📚 Lessons</h2>
    <div class="card-grid" id="lesson-grid">
      {cards_html}
    </div>
  </main>
  <footer>
    Developed by Deshraj Jogiya · Apr 2023 – Mar 2024 ·
    User satisfaction 4.6 / 5 (50+ student surveys)
  </footer>
</body>
</html>"""
    return html


@app.route("/lesson/<name>", methods=["GET"])
def lesson_view(name: str):
    """Full lesson view with description, key ASL terms, objectives, and quiz."""
    lesson = get_lesson_by_name(name)
    if lesson is None:
        return f"<h2>Lesson '{name}' not found.</h2><a href='/'>← Back</a>", 404

    # Key terms
    terms_html = ""
    for t in lesson["key_terms"]:
        terms_html += f"""
        <div class="term-card" id="term-{t['asl_gesture_id']}-{t['term'].replace(' ','_')}">
          <div class="gesture">{t['asl_gesture_id']}</div>
          <div class="term-name">{t['term']}</div>
          <div class="term-def">{t['definition']}</div>
        </div>
        """

    # ASL gesture IDs
    gestures_html = "".join(
        f'<span class="asl-letter" title="ASL: {g}">{g}</span>'
        for g in lesson["asl_gesture_ids"]
    )

    # Learning objectives
    obj_html = "".join(
        f"<li>{obj}</li>" for obj in lesson["learning_objectives"]
    )

    # Quiz
    quiz_html = ""
    for i, q in enumerate(lesson["quiz_questions"], 1):
        options_html = "".join(
            f'<li class="{"correct" if opt == q["answer"] else ""}">{opt}</li>'
            for opt in q["options"]
        )
        quiz_html += f"""
        <div class="quiz-question" id="quiz-q{i}">
          <p>Q{i}. {q['question']}</p>
          <ul class="quiz-options">{options_html}</ul>
          <p style="font-size:0.8rem;color:#8b949e;margin-top:.5rem">
            💡 {q['explanation']}
          </p>
        </div>
        """

    # Related lessons
    related_html = ""
    for rid in lesson.get("related_lessons", []):
        related = next((l for l in LESSONS if l["id"] == rid), None)
        if related:
            rurl = url_for("lesson_view", name=related["name"].lower())
            related_html += f'<a class="card" href="{rurl}" id="related-{rid}" style="padding:.75rem 1rem">{_domain_badge(related["domain"])} <span style="font-weight:600">{related["name"]}</span></a>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{lesson['name']} | STEM × ASL</title>
  <meta name="description" content="Learn {lesson['name']} with ASL gesture support.">
  {BASE_CSS}
</head>
<body>
  {_navbar()}
  <main>
    <a class="back-link" href="/" id="back-to-index">← All Lessons</a>

    {_domain_badge(lesson['domain'])}
    <h1 id="lesson-title">{lesson['name']}</h1>
    <p style="color:{_difficulty_color(lesson['difficulty_level'])};font-size:.85rem;margin-bottom:1.5rem">
      ⬤ {lesson['difficulty_level']}
    </p>

    <!-- Description -->
    <div class="section" id="section-description">
      <h2>📖 Overview</h2>
      <p>{lesson['description']}</p>
      <h3>ASL Gesture Sequence</h3>
      <div class="asl-gestures" id="asl-gestures">{gestures_html}</div>
    </div>

    <!-- Learning Objectives -->
    <div class="section" id="section-objectives">
      <h2>🎯 Learning Objectives</h2>
      <ol class="objectives-list">{obj_html}</ol>
    </div>

    <!-- Key Terms -->
    <div class="section" id="section-terms">
      <h2>🔑 Key Terms & ASL Signs</h2>
      <div class="key-terms-grid">{terms_html}</div>
    </div>

    <!-- Quiz -->
    <div class="section" id="section-quiz">
      <h2>🧠 Quick Quiz</h2>
      <div class="quiz-block">{quiz_html}</div>
    </div>

    <!-- Related Lessons -->
    {"<div class='section' id='section-related'><h2>🔗 Related Lessons</h2><div class='card-grid' style='margin-top:.75rem'>" + related_html + "</div></div>" if related_html else ""}

  </main>
  <footer>
    Extending STEM across ASL · Deshraj Jogiya · 2023–2024
  </footer>
</body>
</html>"""
    return html


@app.route("/search", methods=["POST"])
def search():
    """Keyword search across all 7 STEM concepts."""
    query = request.form.get("query", "").strip()

    results = engine.search(query, deduplicate=True) if query else []

    results_html = ""
    for r in results:
        similar = ""
        if r.get("similar_concepts"):
            similar = f'<div class="similar-badge">Also covers: {", ".join(r["similar_concepts"])}</div>'
        lesson_url = url_for("lesson_view", name=r["name"].lower())
        results_html += f"""
        <div class="search-result-card" id="result-{r['id']}">
          <div style="flex:1">
            {_domain_badge(r['domain'])}
            <a href="{lesson_url}" id="result-link-{r['id']}">{r['name']}</a>
            <p style="font-size:.85rem;color:#8b949e;margin:.3rem 0 0">
              {r['description'][:160]}…
            </p>
            {similar}
          </div>
          <span class="score-pill">score {r['score']}</span>
        </div>
        """

    no_results = ""
    if query and not results:
        no_results = '<p class="no-results">No concepts matched your query. Try terms like "gravity", "DNA", or "algebra".</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Search: {query} | STEM × ASL</title>
  {BASE_CSS}
</head>
<body>
  {_navbar(query=query)}
  <main>
    <a class="back-link" href="/" id="back-to-index">← All Lessons</a>
    <h1>Search Results</h1>
    <p class="subtitle">
      {'Showing results for: <strong>' + query + '</strong>' if query else 'Enter a search term above.'}
    </p>
    <div class="search-results" id="search-results">
      {results_html or no_results}
    </div>
  </main>
  <footer>Extending STEM across ASL · Deshraj Jogiya · 2023–2024</footer>
</body>
</html>"""
    return html


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Gesture prediction endpoint.

    Accepts:
        - multipart/form-data with field 'image' (PNG/JPG file), OR
        - JSON body with field 'image_b64' (base64-encoded image string)

    Returns JSON:
        {
          "letter":      "A",
          "confidence":  0.87,
          "class_index": 0,
          "model_loaded": true/false
        }
    """
    import numpy as np

    # --- Lazy model load (avoid importing TF at module level) -----------------
    try:
        from asl_recognition.predict import load_model, predict, generate_mock_input
        from asl_recognition.model  import get_label_map, IMG_HEIGHT, IMG_WIDTH
        model     = load_model()
        label_map = get_label_map()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        return jsonify({
            "error": f"Model could not be loaded: {str(e)}",
            "model_loaded": False,
        }), 500

    # --- Get image data -------------------------------------------------------
    image_array = None

    if "image" in request.files:
        try:
            from PIL import Image
            file = request.files["image"]
            img  = Image.open(file.stream).convert("L")
            img  = img.resize((IMG_WIDTH, IMG_HEIGHT))
            arr  = np.array(img, dtype=np.float32) / 255.0
            image_array = arr[np.newaxis, :, :, np.newaxis]
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    elif request.is_json and "image_b64" in request.get_json():
        try:
            from PIL import Image
            data     = base64.b64decode(request.get_json()["image_b64"])
            img      = Image.open(io.BytesIO(data)).convert("L")
            img      = img.resize((IMG_WIDTH, IMG_HEIGHT))
            arr      = np.array(img, dtype=np.float32) / 255.0
            image_array = arr[np.newaxis, :, :, np.newaxis]
        except Exception as e:
            return jsonify({"error": f"Base64 image processing failed: {str(e)}"}), 400

    else:
        # Fall back to mock input
        image_array = generate_mock_input(seed=42)

    # --- Predict ---------------------------------------------------------------
    letter, confidence = predict(image_array, model, label_map)

    return jsonify({
        "letter":       letter,
        "confidence":   round(confidence, 4),
        "class_index":  ord(letter) - ord("A"),
        "model_loaded": model_loaded,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Extending STEM across ASL – Flask Web App")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
