import re
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema.messages import HumanMessage, AIMessage

class LangChain:
    CHROMA_PATH = "./chroma/"
    PROMPT_TEMPLATE = """
    Hola, eres un chatbot de Sabentis, una plataforma para la gestión de la seguridad y los riesgos laborales en el trabajo.
    Necesito tu ayuda para responder a una pregunta basándote solamente en información específica que tengo aquí. Aquí está el contexto:

    {context}

    ---

    Teniendo en cuenta solamente la información anterior, ¿podrías ayudarme a responder esto de manera clara y amigable?: {question}
    """

    def __init__(self):
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(persist_directory=self.CHROMA_PATH, embedding_function=self.embedding_function)
        self.conversation = ConversationChain(
            llm=ChatOpenAI(),
            verbose=False,
            memory=ConversationBufferMemory(return_messages=True)
        )
        self.previous_conversation = []

    def is_greeting(self, query):
        greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "saludos"]
        return any(re.match(f"\\b{greet}\\b", query.lower()) for greet in greetings)

    def get_greeting_response(self, query):
        return "¡Hola! ¿En qué puedo ayudarte hoy?"

    def format_response(self, text):
        return re.sub(r"\.\s*", ".\n", text)

    def predict(self, query_text):
        human_message = HumanMessage(content=query_text)

        if self.is_greeting(query_text):
            response_text = self.get_greeting_response(query_text)
            return self.format_response(response_text)

        results = self.db.similarity_search_with_relevance_scores(query_text, k=3)
        filtered_results = [(doc, score) for doc, score in results if score >= 0.75]

        if not filtered_results:
            return "Lo siento, parece ser que no puedo encontrar respuesta a tu pregunta. ¿Podrías formular otra pregunta para poder ayudarte mejor?"

        context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in filtered_results])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=human_message.content)

        response_text = self.conversation.predict(input=prompt)
        ai_message = AIMessage(content=response_text)

        self.previous_conversation.append({"input": human_message.content, "output": ai_message.content})

        sources_similarities = {}
        for doc, score in filtered_results:
            source = doc.metadata.get('source', None)
            if source not in sources_similarities:
                sources_similarities[source] = []
            sources_similarities[source].append(score)

        averaged_sources = {source: sum(similarities) / len(similarities) for source, similarities in sources_similarities.items()}
        
        formatted_sources = "\n".join([f"- {source}\n  {average:.2f}" for source, average in averaged_sources.items()])
        formatted_response = f"Respuesta: {self.format_response(ai_message.content)}\nFuentes y Similitud Promedio:\n{formatted_sources}"
        return formatted_response
