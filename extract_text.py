import os
import fitz  # PyMuPDF
import re

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

def split_text_into_sentences(text):
    sentences = text.split("\n \n")
    return sentences

def clean_and_create_sentences(text):
    # Remove extra spaces and newlines
    text = text.strip()
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")

    # Remove numbers with a + before them
    text = re.sub(r"\+\d+", "", text)

    sentences = split_text_into_sentences(text)
    
    # Remove sentences with less than 10 characters
    sentences = [sent for sent in sentences if len(sent) > 10]

    # Remove sentences with any text with this format xx.xxx.xx.xx
    sentences = [sent for sent in sentences if not re.search(r"\d{2}\.\d{3}\.\d{2}\.\d{2}", sent)]

    # Remove sentences with the format Imagen x:
    sentences = [sent for sent in sentences if not re.search(r"Imagen \d+", sent)]

    # Remove sentences with more than 5 dots in a row:
    sentences = [sent for sent in sentences if not re.search(r"\.{5,}", sent)]

    # Remove sentences that start with a number and have less than 30 characters
    sentences = [sent for sent in sentences if not re.search(r"^\d", sent) or len(sent) > 30]

    # Remove sentences that start with Manual de usuario: or Manual de usuario
    sentences = [sent for sent in sentences if not re.search(r"Manual de usuario".lower(), sent.lower())]

    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("\n", " ")
        sentences[i] = sentences[i].strip()

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
