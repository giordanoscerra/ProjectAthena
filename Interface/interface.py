import os
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import joblib

# Path to the PDF file
pdf_file = "/Users/davide/Documents/GitHub/ProjectAthena/Interface/hlt.pdf"

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

print(predictions)

# Output the colored text to a file
output_file = "Interface/output.html"

# Create the output file
with open(output_file, "w") as file:
    # Write the HTML header
    file.write("<html><head><title>HLT Output</title></head><body>")
    
    # Write the text with color-coded class predictions
    for sentence, prediction in zip(sentences, predictions):
        color_code = hash(prediction[0]) % 16777215  # Generate a color code based on the hash of the prediction
        color = f"#{color_code:06x}"  # Convert the color code to hexadecimal format
        file.write(f"<p style='color:{color}'>{sentence}</p>")
    
    # Write the HTML footer
    file.write("</body></html>")

# Open the output file in the default web browser
os.system(f"open {output_file}")
