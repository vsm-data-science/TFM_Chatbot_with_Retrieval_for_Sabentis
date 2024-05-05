import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "./chroma/"
PROMPT_TEMPLATE = """
Hola, eres un chatbot de Sabentis, una plataforma para la gestión de la seguridad y los riesgos laborales en el trabajo.
Necesito tu ayuda para responder a una pregunta basándome solamente en información específica que tengo aquí. Aquí está el contexto:

{context}

---

Teniendo en cuenta sólo la información anterior, ¿podrías ayudarme a responder esto de manera clara y amigable?: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # Filtrar resultados con un score menor a 0.7
    filtered_results = [(doc, score) for doc, score in results if score >= 0.7]
    
    if not filtered_results:
        print("Vaya, parece ser que no puedo encontrar respuesta a tu pregunta. ¿Podrías formular otra pregunta para poder ayudarte mejor?")
        return  # Salir si no hay resultados adecuados

    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    # Crear instancia del modelo
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    
    # Incluir similarity scores en la respuesta para referencia
    scores = [f"{doc.metadata.get('source', None)}: {score:.2f}" for doc, score in filtered_results]
    formatted_response = f"Respuesta: {response_text}\nSimilitud y Fuentes: {scores}"
    print(formatted_response)

if __name__ == "__main__":
    main()
