import os
import fitz  # PyMuPDF
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize a variable to hold all the extracted text
    text = ""
    # Iterate over each page and extract text
    for page in doc:
        text += page.get_text()
    
    # Close the document
    doc.close()
    
    return text

def write_text_to_file(text, output_path):
    with open(output_path, "w") as file:
        file.write(text)

def clean_and_create_sentences(text):
    # Remove extra spaces and newlines
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")

    # Remove numbers with a + before them
    text = re.sub(r"\+\d+", "", text)
    
    # Use spacy to create sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # # Remove sentences with less than 5 characters
    sentences = [sent for sent in sentences if len(sent) > 5]

    # # Remove sentences with any text with this format xx.xxx.xx.xx
    sentences = [sent for sent in sentences if not re.search(r"\d{2}\.\d{3}\.\d{2}\.\d{2}", sent)]

    # # Remove sentences with the format Imagen x:
    sentences = [sent for sent in sentences if not re.search(r"Imagen \d+:", sent)]

    # # Remove sentences with more than 5 dots in a row:
    sentences = [sent for sent in sentences if not re.search(r"\.{5,}", sent)]

    # # Remove sentences with the following text Todos los derechos reservados EASY TECH GLOBAL:
    sentences = [sent for sent in sentences if not re.search(r"Todos los derechos reservados EASY TECH GLOBAL", sent)]

    return sentences

# Get all files in pdfs directory
pdfs_dir = "pdfs"
pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]

# Create full pdf path by appending the folder
pdf_files = [os.path.join(pdfs_dir, f) for f in pdf_files]

# Iterate over each PDF file and extract text
for pdf_file in pdf_files:
    # Create txt folder if it does not exist
    txt_dir = "txt"
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    text = extract_text_from_pdf(pdf_file)
    sentences = clean_and_create_sentences(text)
    text = "\n\n".join(sentences)
    output_path = pdf_file.replace(".pdf", ".txt").replace("pdfs", "txt")
    write_text_to_file(text, output_path)
