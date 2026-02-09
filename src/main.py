# ==============================
# IMPORT REQUIRED LIBRARIES
# ==============================

import os
from datetime import date
from flask import Flask, render_template, request
from transformers import pipeline


# ==============================
# FLASK APP CONFIGURATION
# ==============================

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


# ==============================
# LOAD AI MODEL (LOADS ONCE)
# ==============================

# Supported models:
# "facebook/bart-large-cnn"  -> Good for long documents
# "google/pegasus-xsum"      -> Better for short summaries

summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=0   # Uses Apple MPS (Mac GPU) if available
)


# ==============================
# ROUTE: HOME PAGE
# ==============================

@app.route("/")
def index():
    """
    Renders the landing page
    """
    return render_template("index.html")


# ==============================
# ROUTE: INPUT PAGE
# ==============================

@app.route("/input")
def input_page():
    """
    Renders the page where user enters notes
    """
    return render_template("input.html")


# ==============================
# ROUTE: GENERATE SUMMARY
# ==============================

@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Receives user input text, processes it using
    Transformer-based AI model, and returns summary
    """

    # Get text entered by user
    original_text = request.form.get("notes", "").strip()

    # ------------------------------
    # SHORT TEXT HANDLING (IMPORTANT)
    # ------------------------------
    # BART & PEGASUS need sufficient context.
    # If text is too short, avoid hallucination.
    if len(original_text.split()) < 30:
        summary_text = (
            "The provided text is too short to generate a meaningful summary. "
            "Please enter more detailed notes."
        )
    else:
        # Generate summary using AI model
        summary_output = summarizer(
            original_text,
            max_length=150,
            min_length=60,
            do_sample=False
        )

        # Extract summary text
        summary_text = summary_output[0]["summary_text"]

    # Get today's date for footer display
    today = date.today().strftime("%d %B %Y")

    # Render result page
    return render_template(
        "result.html",
        original_text=original_text,
        summary=summary_text,
        today=today
    )


# ==============================
# RUN FLASK APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=True)