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

# Get base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


# ==============================
# LOAD AI MODEL (ONCE)
# ==============================

# You can switch model here:
# "facebook/bart-large-cnn"
# "google/pegasus-xsum"

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0   # Uses Apple MPS if available
)


# ==============================
# ROUTE: HOME PAGE
# ==============================

@app.route("/")
def index():
    return render_template("index.html")


# ==============================
# ROUTE: INPUT PAGE
# ==============================

@app.route("/input")
def input_page():
    return render_template("input.html")


# ==============================
# ROUTE: PROCESS SUMMARY
# ==============================

@app.route("/summarize", methods=["POST"])
def summarize():

    # Get text from form textarea
    original_text = request.form["notes"]

    # Generate AI summary
    summary_output = summarizer(
        original_text,
        max_length=150,
        min_length=60,
        do_sample=False
    )

    summary_text = summary_output[0]["summary_text"]

    today = date.today().strftime("%d %B %Y")

    return render_template(
        "result.html",
        original_text=original_text,
        summary=summary_text,
        today=today
    )


# ==============================
# RUN FLASK APP
# ==============================

if __name__ == "__main__":
    app.run(debug=True)