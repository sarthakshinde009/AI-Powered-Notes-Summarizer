#from transformers import pipeline
#
#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#
#print("AI Model Loaded!")




from transformers import pipeline

# Load the AI summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("AI Model Loaded!")

# Sample text to summarize
text = "Artificial Intelligence is transforming the world by automating tasks, improving decision-making, and enabling new technologies."

# Generate summary
summary = summarizer(text, max_length=30, min_length=5, do_sample=False)

# Print the result
print("Summary:", summary[0]["summary_text"])