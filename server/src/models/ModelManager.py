import os
from src.utils.FileProcessor import FileProcessor
from src.models.BERTModel import BERTModel
from src.models.PretrainedModel import PretrainedModel
from src.models.TFIDFModel import TFIDFModel
from src.models.Word2VecModel import Word2VecModel
from src.models.Word2VecTFIDFModel import Word2VecTFIDFModel
from openai import OpenAI

client = OpenAI()

class ModelManager:
    def __init__(self, file_processor, pretrained_model_path=None):
        self.models = {
            # "word2vec": Word2VecModel(file_processor.original_paragraphs, file_processor.language),
            # "tfidf": TFIDFModel(file_processor.preprocessed_paragraphs, file_processor.original_paragraphs, file_processor.language),
            # "word2vec_tfidf": Word2VecTFIDFModel(file_processor.original_paragraphs,
            #                                       file_processor.preprocessed_paragraphs,
            #                                       file_processor.language),
            "bert": BERTModel(file_processor.original_paragraphs, file_processor.language)
        }

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.models["pretrained"] = PretrainedModel(file_processor.original_paragraphs, file_processor.language, pretrained_model_path)

    def find_most_similar_sentences(self, query, similarity_threshold=0.55):
        results = []
        for name, model in self.models.items():
            try:
                model_results = model.predict(query, similarity_threshold)
                if model_results:
                    results.extend(model_results[:5])
            except Exception as e:
                print(f"Error al procesar el modelo {name}: {e}")

        return results
    
    # TODO: Initialize the model in a different function
    @staticmethod
    def find_most_similar_dictionary(query):
        highest_similarity_score = 0.0
        file_with_highest_similarity = None
        results_with_high_similarity = []

        pretrained_model_path = 'sbw_vectors.bin'  
        txt_directory = 'txt'
        for filename in os.listdir(txt_directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(txt_directory, filename)
                file_processor = FileProcessor(filepath, 'spanish')
                model_manager = ModelManager(file_processor, pretrained_model_path)

                similarities = model_manager.find_most_similar_sentences(query)
                paragraphs = [paragraph for _, paragraph in similarities]

                for similarity, _ in similarities:
                    if similarity > highest_similarity_score:
                        highest_similarity_score = similarity
                        file_with_highest_similarity = filename
                        results_with_high_similarity = paragraphs

        results_with_high_similarity = list(set(results_with_high_similarity))
        return file_with_highest_similarity, results_with_high_similarity
    
    @staticmethod
    def get_chat_gpt_answer(file, texts, query):
        all_texts = ", ".join(texts)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", "content": 
                        f"""
                        Eres un asistente de la plataforma Sabentis, que ayuda a los usuarios a encontrar información sobre la seguridad y salud en el trabajo.
                        Siempre que un usuario haga una pregunta te voy a pasar algunos textos que vienen desde sus manuales para ayudarte a tener un poco mas de contexto.
                        El siguiente texto es una introduccion a la plataforma:
                        Bienvenido a Sabentis,
                        ¿Qué es Sabentis? Es un software de gestión para la seguridad y salud en el trabajo, que permite automatizar, notificar y centralizar toda la información del SG-SST de la organización. La plataforma sigue las directrices de la ISO 45001 y de la Organización Internacional del Trabajo (OIT) y ofrece una experiencia de usuario completa. 
                        Ventajas de utilizar Sabentis. Proporciona una intuitiva experiencia al usuario con tecnología avanzada para una gestión de la SST eficaz, segura y adaptable. Sabentis está impulsada por las últimas tecnologías de programación, que lo convierten en un sistema hiperversátil que se adapta a todo tipo de dispositivos. Además, automatiza tareas repetitivas, ahorrando tiempo y recursos y mejora la comunicación entre los trabajadores y la empresa mediante sistemas de alerta y notificaciones.
                        Recuerda siempre decirle al usuario el nombre del archivo del que estas extrayendo la información y el contenido de este.
                        En este caso es {file} y los textos son: {all_texts}
                        """
                        },
                        {"role": "user", "content": query}
                    ]
        )
        return completion.choices[0].message.content
