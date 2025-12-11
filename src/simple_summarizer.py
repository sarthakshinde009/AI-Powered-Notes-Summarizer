#from transformers import pipeline
#
## ---------- SET YOUR FILE HERE ----------
#filename = "data/doc2.txt"   # ← change this to note2.txt or note3.txt
## ----------------------------------------
#
## Read the file
#with open(filename, "r", encoding="utf-8") as f:
#    text = f.read()
#
## Load summarizer model
#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#
#print("AI Model Loaded!")
#print("Summarizing...\n")
#
## Generate summary
#summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
#
## Print summary
#print("📘 AI Summary:")
#print(summary[0]["summary_text"])




from transformers import pipeline

# ---------- SETTINGS ----------
filename = "data/doc5.txt"     # change note1 → note2 → note3
max_words = 80                  # ← control summary length here
# ------------------------------

# Read the file
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# Load the model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("AI Model Loaded!")
print("Summarizing...\n")

# Generate summary
summary = summarizer(
    text,
    max_length=max_words,
    min_length=int(max_words * 0.4),   # keeps summary balanced
    do_sample=False
)

# Print summary
print("📘 AI Summary:")
print(summary[0]["summary_text"])