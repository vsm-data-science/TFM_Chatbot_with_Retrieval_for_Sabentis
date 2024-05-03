from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

CHROMA_PATH = "./chroma/"
DATA_PATH = "./txt/"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

    # Intentar recuperar un documento para verificación
    test_query = "24 EVALUACIÓN DE ASPECTOS LEGALES"
    results = db.similarity_search(test_query, k=1)
    print("Resultados:", results)  # Imprime la estructura completa de los resultados para verificar
    if results:
        print("Verificación del contenido guardado exitosamente:")
        for result in results:
            doc = result  # El resultado es un Documento directamente
            print(f"Contenido del documento: {doc.page_content[:100]}")
    else:
        print("No se encontraron documentos durante la verificación.")

if __name__ == "__main__":
    main()
