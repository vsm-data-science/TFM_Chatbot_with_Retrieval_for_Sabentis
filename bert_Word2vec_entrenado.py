import os
import numpy as np
import unidecode
import re
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Asegúrate de haber descargado los paquetes necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

class FileProcessor:
    def __init__(self, filepath, language='spanish'):
        self.filepath = filepath
        self.language = language
        self._verify_filepath()

    def _verify_filepath(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"El archivo especificado no existe: {self.filepath}")

    def _clean_and_tokenize(self, text):
        text = unidecode.unidecode(text)
        stop_words = set(stopwords.words(self.language))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        tokens = word_tokenize(text, language=self.language)
        return [token for token in tokens if token not in stop_words]

    def read_and_preprocess(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        paragraphs = content.split('\n\n')
        paragraphs = [para.strip() for para in paragraphs if para.strip() != '']
        preprocessed_paragraphs = [' '.join(self._clean_and_tokenize(para)) for para in paragraphs]
        return paragraphs, preprocessed_paragraphs

class ModelManager:
    def __init__(self, original_paragraphs, preprocessed_paragraphs, word2vec_model_path, bert_model_name='dccuchile/bert-base-spanish-wwm-cased'):
        self.original_paragraphs = original_paragraphs
        self.preprocessed_paragraphs = preprocessed_paragraphs
        self.word2vec_model = self._load_pretrained_word2vec_model(word2vec_model_path)
        self.bert_tokenizer, self.bert_model = self._load_bert_model(bert_model_name)
        self.tfidf_matrix, self.tfidf_feature_names = self._train_tfidf_model(preprocessed_paragraphs)

    def _load_pretrained_word2vec_model(self, model_path):
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        return model

    def _load_bert_model(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        return tokenizer, model

    def _train_tfidf_model(self, preprocessed_paragraphs):
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='spanish')
        tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_paragraphs)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        return tfidf_matrix, tfidf_feature_names

    def _get_word2vec_embedding(self, text):
        words = text.split()
        word_vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model.key_to_index]
        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(word_vectors, axis=0)

    def _get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled.detach().numpy()

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0
        return 1 - spatial.distance.cosine(vec1, vec2)

    def find_most_similar_sentences_word2vec_pretrained(self, query):
        query_embedding = self._get_word2vec_embedding(query)
        similarities = []

        for paragraph in self.preprocessed_paragraphs:
            paragraph_embedding = self._get_word2vec_embedding(paragraph)
            similarity = self._cosine_similarity(query_embedding, paragraph_embedding)
            similarities.append((similarity, paragraph))

        # Ordena las similitudes de mayor a menor
        sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        # Devuelve las dos oraciones más similares
        return sorted_similarities[:2]

filepath = r'C:\Users\vsanc\OneDrive\Escritorio\Proyecto ChatBot Sabentis TFM\sabentis_recommendation_system-main\txt\MANUAL AUDITORIAS_ESP (2)'
word2vec_model_path = 'ruta_al_modelo_preentrenado_word2vec.bin'

try:
    file_processor = FileProcessor(filepath)
    original_paragraphs, preprocessed_paragraphs = file_processor.read_and_preprocess()
    model_manager = ModelManager(original_paragraphs, preprocessed_paragraphs, word2vec_model_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Introduce una consulta para encontrar los párrafos más similares
query = "Introduce tu consulta aquí"

# Encuentra y muestra los resultados usando Word2Vec preentrenado
print("Resultados Word2Vec Preentrenado:")
similarities_w2v_pretrained = model_manager.find_most_similar_sentences_word2vec_pretrained(query)
for i, (similarity, paragraph) in enumerate(similarities_w2v_pretrained, start=1):
    print(f"{i}. Similitud: {similarity:.4f}, Párrafo: {paragraph}\n")

        
        