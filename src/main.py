from transformers import pipeline
from html import escape
from pathlib import Path
from datetime import date

# -----------------------------
# CONFIGURATION
# -----------------------------
NOTE_FILE = "data/doc2.txt"
MAX_WORDS = 80
OUTPUT_HTML = "ui/summary_report.html"
AUTHOR_NAME = "Sarthak Shinde"
TODAY_DATE = date.today().strftime("%d %B %Y")
MODEL_NAME = "facebook/bart-large-cnn"

# -----------------------------
# LOAD AI MODEL
# -----------------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# -----------------------------
# READ INPUT NOTE
# -----------------------------
with open(NOTE_FILE, "r", encoding="utf-8") as f:
    original_text = f.read()

# -----------------------------
# GENERATE SUMMARY
# -----------------------------
summary = summarizer(
    original_text,
    max_length=MAX_WORDS,
    min_length=30,
    do_sample=False
)[0]["summary_text"]

# -----------------------------
# ESCAPE TEXT FOR HTML
# -----------------------------
orig_esc = escape(original_text)
sum_esc = escape(summary)

# -----------------------------
# CREATE HTML (INLINE CSS)
# -----------------------------
html_page = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>My AI Notes Summarizer</title>
</head>

<body style="margin:0; font-family:Arial, Helvetica, sans-serif; background-color:#0d1224; color:white;">

    <h1 style="text-align:center; margin:30px 0;">
        My AI Notes Summarizer
    </h1>

    <div style="
        max-width:1100px;
        margin:0 auto;
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:30px;
    ">

        <!-- Original Text -->
        <div style="
            border:2px solid #1e90ff;
            background-color:#0b1d3a;
            padding:20px;
            border-radius:12px;
        ">
            <h2 style="color:#8ab4ff;">Original Text</h2>
            <p style="color:#dbeafe; white-space:pre-wrap;">
                {orig_esc}
            </p>
        </div>

        <!-- AI Summary -->
        <div style="
            border:2px solid #22c55e;
            background-color:#052e16;
            padding:20px;
            border-radius:12px;
        ">
            <h2 style="color:#86efac;">AI Summary</h2>
            <p style="color:#dcfce7; white-space:pre-wrap;">
                {sum_esc}
            </p>
        </div>

    </div>
        <div style="
        margin:40px 0 20px 0;
        text-align:center;
        font-size:14px;
        color:#94a3b8;
    ">
        Summary created by <strong>{AUTHOR_NAME}</strong>
        on <strong>{TODAY_DATE}</strong>
        using <strong>{MODEL_NAME}</strong> model
    </div>

</body>
</html>
"""

# -----------------------------
# SAVE HTML OUTPUT
# -----------------------------
output_path = Path(OUTPUT_HTML)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_page)

print("✅ Summary generated and saved as ui/summary_report.html")