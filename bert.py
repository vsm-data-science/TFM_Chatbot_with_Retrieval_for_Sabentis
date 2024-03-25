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
    with open(output_path, "w", encoding="utf-8") as file:
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
pdfs_dir = r"C:\Users\vsanc\OneDrive\Escritorio\Proyecto ChatBot Sabentis TFM\sabentis_recommendation_system-main\pdfs"
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

from transformers import BertTokenizer, BertModel
import torch
from scipy import spatial
import numpy as np

class BertEmbeddings:
    def __init__(self, bert_model_name='dccuchile/bert-base-spanish-wwm-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled[0].numpy()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0
        return 1 - spatial.distance.cosine(vec1, vec2)

# Ejemplo de uso
bert_embeddings = BertEmbeddings()

# Obtiene embeddings para dos textos de ejemplo
text1_embedding = bert_embeddings.get_embedding("Cómo generar una documentación")
text2_embedding = bert_embeddings.get_embedding("generar el informe de la evaluación")

# Calcula y muestra la similitud del coseno entre los dos embeddings
similarity = bert_embeddings.cosine_similarity(text1_embedding, text2_embedding)
print(f"Similitud del coseno: {similarity}")
