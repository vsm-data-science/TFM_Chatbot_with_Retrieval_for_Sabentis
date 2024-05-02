import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "./chroma/"
PROMPT_TEMPLATE = """
Responde a la pregunta basado solamente en el siguiente contexto:

{context}

---

Responde a la pregunta basado solamente en el contexto de arriba: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print("Sin resultados")
    elif results[0][1] < 0.3:
        print("Resultados menores a 0.3")
    else:
        print("Unable to find matching results.")
    return
    
    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    model = ChatOpenAI
    response_text = model.predict(prompt)
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Respuesta: {response_text}\nFuente: {sources}"
    print(formatted_response)
    
if __name__ == "__main__":
    main()