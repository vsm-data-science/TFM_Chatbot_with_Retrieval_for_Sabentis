import os
import fitz  # PyMuPDF

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
    # print(text)
    output_path = pdf_file.replace(".pdf", ".txt").replace("pdfs", "txt")
    print(output_path)
    write_text_to_file(text, output_path)
