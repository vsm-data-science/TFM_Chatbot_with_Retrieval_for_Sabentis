import os
import re
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from transformers import BertTokenizer, BertModel
import torch
from openai import OpenAI
from flask import Flask, request, jsonify

client = OpenAI()

def download_nltk_packages():
    packages = ['stopwords', 'punkt']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_packages()

class TextUtils:
    @staticmethod
    def clean_and_tokenize(text, language):
        text = unidecode.unidecode(text.lower())
        stop_words = set(stopwords.words(language))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        tokens = word_tokenize(text, language=language)
        return [token for token in tokens if token not in stop_words]


class FileProcessor:
    def __init__(self, filepath, language='spanish'):
        self.filepath = filepath
        self.language = language
        self.original_paragraphs, self.preprocessed_paragraphs = self.read_and_preprocess()

    def _verify_filepath(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"El archivo especificado no existe: {self.filepath}")

    def read_and_preprocess(self):
        self._verify_filepath()
        with open(self.filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        paragraphs = content.split('\n\n')
        original_paragraphs = [para.strip() for para in paragraphs if para.strip() != '']
        preprocessed_paragraphs = [' '.join(TextUtils.clean_and_tokenize(para, self.language)) for para in original_paragraphs]
        return original_paragraphs, preprocessed_paragraphs

class Model:
    def __init__(self, original_paragraphs, preprocessed_paragraphs, language='spanish'):
        self.original_paragraphs = original_paragraphs
        self.preprocessed_paragraphs = preprocessed_paragraphs
        self.language = language

    def predict(self, query, similarity_threshold):
        raise NotImplementedError("Subclasses should implement this method")
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return -1
        return 1 - spatial.distance.cosine(vec1, vec2)
    
class Word2VecModel(Model):
    def __init__(self, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, [], language)
        self.model = self._train_model()

    def _train_model(self):
        sentences = [TextUtils.clean_and_tokenize(paragraph, self.language) for paragraph in self.original_paragraphs]
        return Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def _get_sentence_embedding(self, sentence):
        words = TextUtils.clean_and_tokenize(sentence, self.language)
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def predict(self, query, similarity_threshold):
        query_embedding = self._get_sentence_embedding(query)
        similarities = [(self._cosine_similarity(query_embedding, self._get_sentence_embedding(' '.join(TextUtils.clean_and_tokenize(para, self.language)))), para) for para in self.original_paragraphs]
        filtered_and_sorted = sorted([sim for sim in similarities if sim[0] > similarity_threshold], key=lambda x: x[0], reverse=True)
        return filtered_and_sorted


class TFIDFModel(Model):
    def __init__(self, preprocessed_paragraphs, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, preprocessed_paragraphs, language)
        self.model, self.feature_names = self._train_model()

    def _train_model(self):
        vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language))
        tfidf_matrix = vectorizer.fit_transform(self.preprocessed_paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        return tfidf_matrix, feature_names

    def predict(self, query, similarity_threshold):
        query_vec = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language), vocabulary=self.feature_names).fit_transform([query])
        cosine_similarities = linear_kernel(query_vec, self.model).flatten()

        similarities = []
        for i, score in enumerate(cosine_similarities):
            if score > similarity_threshold:
                if i < len(self.original_paragraphs):
                    similarities.append((score, self.original_paragraphs[i]))

        filtered_and_sorted = sorted(similarities, key=lambda x: x[0], reverse=True)
        return filtered_and_sorted

    
class Word2VecTFIDFModel(Model):
    def __init__(self, original_paragraphs, preprocessed_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, preprocessed_paragraphs, language)
        self.word2vec_model = Word2VecModel(original_paragraphs, language)
        self.tfidf_model = TFIDFModel(preprocessed_paragraphs, language)

    def predict(self, query, similarity_threshold):
        query_embedding_w2v = self.word2vec_model._get_sentence_embedding(query)
        query_vec_tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language),
                                          vocabulary=self.tfidf_model.feature_names).fit_transform([' '.join(TextUtils.clean_and_tokenize(query, self.language))])
        similarities = []
        for i, paragraph in enumerate(self.original_paragraphs):
            paragraph_embedding_w2v = self.word2vec_model._get_sentence_embedding(paragraph)
            similarity_w2v = self._cosine_similarity(query_embedding_w2v, paragraph_embedding_w2v)

            similarity_tfidf = linear_kernel(query_vec_tfidf, self.tfidf_model.model[i:i+1]).flatten()[0]
            combined_similarity = (similarity_w2v + similarity_tfidf) / 2

            if combined_similarity > similarity_threshold:
                similarities.append((combined_similarity, paragraph))

        filtered_and_sorted = sorted(similarities, key=lambda x: x[0], reverse=True)
        return filtered_and_sorted

    def _get_sentence_embedding(self, sentence):
        raise NotImplementedError("This method is not used in Word2VecTFIDFModel")

class BERTModel(Model):
    def __init__(self, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, [], language)
        self.model, self.tokenizer = self._load_bert_model()

    def _load_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        return model, tokenizer

    def predict(self, query, similarity_threshold):
        query_embedding = self._get_bert_embedding(query)
        similarities = [(self._cosine_similarity(query_embedding, self._get_bert_embedding(paragraph)), paragraph) for paragraph in self.original_paragraphs]
        filtered_and_sorted = sorted([sim for sim in similarities if sim[0] > similarity_threshold], key=lambda x: x[0], reverse=True)
        return filtered_and_sorted

    def _get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

class PretrainedModel(Model):
    def __init__(self, original_paragraphs, language='spanish', model_path='sbw_vectors.bin'):
        super().__init__(original_paragraphs, [], language)
        self.model = self._load_pretrained_model(model_path)

    def _load_pretrained_model(self, model_path):
        if os.path.exists(model_path):
            return KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
            raise FileNotFoundError(f"El modelo preentrenado no se encuentra en la ruta: {model_path}")

    def predict(self, query, similarity_threshold):
        query_embedding = self._get_sentence_embedding_pretrained(query)
        similarities = [(self._cosine_similarity(query_embedding, self._get_sentence_embedding_pretrained(' '.join(TextUtils.clean_and_tokenize(para, self.language)))), para) for para in self.original_paragraphs]
        filtered_and_sorted = sorted([sim for sim in similarities if sim[0] > similarity_threshold], key=lambda x: x[0], reverse=True)
        return filtered_and_sorted

    def _get_sentence_embedding_pretrained(self, sentence):
        words = TextUtils.clean_and_tokenize(sentence, self.language)
        word_vectors = [self.model[word] for word in words if word in self.model]
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

class ModelManager:
    def __init__(self, file_processor, pretrained_model_path=None):
        self.models = {
            "word2vec": Word2VecModel(file_processor.original_paragraphs, file_processor.language),
            "tfidf": TFIDFModel(file_processor.preprocessed_paragraphs, file_processor.original_paragraphs, file_processor.language),
            "word2vec_tfidf": Word2VecTFIDFModel(file_processor.original_paragraphs,
                                                  file_processor.preprocessed_paragraphs,
                                                  file_processor.language),
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
    
    @staticmethod
    def find_most_similar_dictionary(query):
        pretrained_model_path = 'sbw_vectors.bin'  
        txt_directory = 'txt'

        highest_similarity_score = 0.0
        file_with_highest_similarity = None
        results_with_high_similarity = []

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

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']
    
    file, results = ModelManager.find_most_similar_dictionary(query)
    chat_gpt_answer = ModelManager.get_chat_gpt_answer(file, results, query)
    
    return jsonify({
        'chat_gpt_answer': chat_gpt_answer,
        'file': file,
        'results': results
    })

if __name__ == '__main__':
    app.run()
