import os
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import joblib
import hashlib
import glob

def getColor(input_string):
    return int(hashlib.sha256(input_string.encode()).hexdigest(), 16) % 16777215

# Find the first PDF file in the folder
pdf_files = glob.glob("Interface/*.pdf")
if pdf_files:
    pdf_file = pdf_files[0]
else:
    raise FileNotFoundError("No PDF files found in the folder")

# Open the PDF file
with open(pdf_file, "rb") as file:
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)

    # Extract text from each page
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

# Split the text into sentences
sentences = sent_tokenize(text)

# Load the classifier model
model_file = "NaiveBayes/naiveBayes.pkl"
classifier = joblib.load(model_file)

vectorizer_file = "NaiveBayes/vectorizer.pkl"
vectorizer = joblib.load(vectorizer_file)

# Predict the class for each sentence
predictions = []
for sentence in sentences:
    prediction = classifier.predict(vectorizer.transform([sentence]))
    predictions.append(prediction)

# Output the colored text to a file
output_file = "Interface/output.html"

# Create the output file
with open(output_file, "w") as file:
    # Write the HTML header
    file.write("<html><head><title>HLT Output</title></head><body>")
    
    # Write the text with color-coded class predictions
    for sentence, prediction in zip(sentences, predictions):
        color_code = getColor(prediction[0].item())  # Generate a color code based on the hash of the prediction
        color = f"#{color_code:06x}"  # Convert the color code to hexadecimal format
        file.write(f"<p style='color:{color}'>{sentence}</p>")
    
    # Write the HTML footer
    file.write("</body></html>")

# Open the output file in the default web browser
os.system(f"open {output_file}")

import matplotlib.pyplot as plt

# Count the occurrences of each class
class_counts = {}
for prediction in predictions:
    class_counts[prediction[0].item()] = class_counts.get(prediction[0].item(), 0) + 1

# Get the class labels and their corresponding counts
labels = list(class_counts.keys())
counts = list(class_counts.values())



# Generate color codes for each class based on the hash of the prediction
colors = [f"#{getColor(label):06x}" for label in labels]

# Plot the pie chart
plt.subplot(1, 2, 1)
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
plt.title('Class Distribution')

# Plot the histogram
plt.subplot(1, 2, 2)
plt.bar(labels, counts, color=colors)
plt.xticks(rotation=45, ha='right')
plt.title('Class Frequencies')

# Display the charts
plt.show()