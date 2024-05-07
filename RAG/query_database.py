from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

CHROMA_PATH = "./chroma/"
PROMPT_TEMPLATE = """
Hola, eres un chatbot de Sabentis, una plataforma para la gestión de la seguridad y los riesgos laborales en el trabajo.
Necesito tu ayuda para responder a una pregunta basándote solamente en información específica que tengo aquí. Aquí está el contexto:

{context}

---

Teniendo en cuenta solamente la información anterior, ¿podrías ayudarme a responder esto de manera clara y amigable?: {question}
"""

def main():
    print("Hola, soy el Chatbot de la plataforma Sabentis, puedes preguntarme cualquier cosa sobre la web. ¿En qué puedo ayudarte?")
    
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    conversation = ConversationChain(
        llm=ChatOpenAI(),
        verbose=False,
        memory=ConversationBufferMemory(return_messages=True)
    )
    previous_conversation = []

    while True:
        query_text = input("Introduce tu pregunta: ")

        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        filtered_results = [(doc, score) for doc, score in results if score >= 0.75]

        if not filtered_results:
            print("Lo siento, parece ser que no puedo encontrar respuesta a tu pregunta. ¿Podrías formular otra pregunta para poder ayudarte mejor?")
            continue 

        context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in filtered_results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        #print(prompt)

        response_text = conversation.predict(input=prompt)
        
        previous_conversation.append({"input": query_text, "output": response_text})

        #print("Sabentis Chatbot:", response_text)
        
        sources_similarities = {}
        for doc, score in filtered_results:
            source = doc.metadata.get('source', None)
            if source not in sources_similarities:
                sources_similarities[source] = []
            sources_similarities[source].append(score)

        averaged_sources = {source: sum(similarities) / len(similarities) for source, similarities in sources_similarities.items()}

        formatted_sources = "\n".join([f"- {source}\n  {average:.2f}" for source, average in averaged_sources.items()])
        formatted_response = f"Respuesta: {response_text}\nFuentes y Similitud Promedio:\n{formatted_sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()