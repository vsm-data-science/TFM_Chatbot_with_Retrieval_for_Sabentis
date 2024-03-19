import os
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk

def download_nltk_packages():
    packages = ['stopwords', 'punkt']
    for package in packages:
        try:
            # Intenta cargar el paquete para verificar si ya está instalado
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            # Si el paquete no está instalado, descárgalo sin producir salidas verbosas
            nltk.download(package, quiet=True)

download_nltk_packages()

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
    def __init__(self, original_paragraphs, preprocessed_paragraphs, language='spanish'):
        self.language = language
        self.original_paragraphs = original_paragraphs
        self.preprocessed_paragraphs = preprocessed_paragraphs
        self.word2vec_model = self._train_word2vec_model()
        self.tfidf_matrix, self.tfidf_feature_names = self._train_tfidf_model()

    def _train_word2vec_model(self):
        sentences = [sentence.split() for sentence in self.preprocessed_paragraphs]
        return Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def _train_tfidf_model(self):
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language))
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        return tfidf_matrix, tfidf_feature_names

    def _get_sentence_embedding(self, sentence, model):
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return -1
        return 1 - spatial.distance.cosine(vec1, vec2)

    def find_most_similar_sentences_word2vec(self, query):
        query_embedding = self._get_sentence_embedding(query, self.word2vec_model)
        top_similarities = [(float('-inf'), ""), (float('-inf'), "")]
        for paragraph in self.preprocessed_paragraphs:
            paragraph_embedding = self._get_sentence_embedding(paragraph, self.word2vec_model)
            similarity = self._cosine_similarity(query_embedding, paragraph_embedding)
            if similarity > top_similarities[0][0]:
                top_similarities[0] = (similarity, paragraph)
                top_similarities.sort(reverse=True)
        top_similarities = [(max(0, sim), para) for sim, para in top_similarities]
        return top_similarities[:2]

    def find_most_similar_sentences_tfidf(self, query):
        query_vec = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language), vocabulary=self.tfidf_feature_names).fit_transform([query])
        cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_similarities_indices = cosine_similarities.argsort()[:-3:-1]
        return [(cosine_similarities[i], self.original_paragraphs[i]) for i in top_similarities_indices]
    
    def find_most_similar_sentences_word2vec_tfidf(self, query):
        # Preprocesa la consulta y obtiene su embedding Word2Vec
        query_tokens = ' '.join(file_processor._clean_and_tokenize(query))
        query_embedding = self._get_sentence_embedding(query_tokens, self.word2vec_model)
        
        # Calcula los vectores TF-IDF para la consulta
        query_tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language), vocabulary=self.tfidf_feature_names).fit_transform([query_tokens])
        
        # Combinación de Word2Vec y TF-IDF
        top_similarities = [(float('-inf'), ""), (float('-inf'), "")]
        for i, paragraph in enumerate(self.preprocessed_paragraphs):
            paragraph_embedding = self._get_sentence_embedding(paragraph, self.word2vec_model)
            
            # Calcula la similitud coseno basada en Word2Vec
            similarity_w2v = self._cosine_similarity(query_embedding, paragraph_embedding)
            
            # Obtiene la similitud TF-IDF
            similarity_tfidf = linear_kernel(query_tfidf, self.tfidf_matrix[i:i+1]).flatten()[0]
            
            # Combina las similitudes (puedes experimentar con diferentes enfoques de combinación)
            combined_similarity = (similarity_w2v + similarity_tfidf) / 2
            
            if combined_similarity > top_similarities[0][0]:
                top_similarities[0] = (combined_similarity, self.original_paragraphs[i])
                top_similarities.sort(reverse=True)
                
        top_similarities = [(max(0, sim), para) for sim, para in top_similarities]
        return top_similarities[:2]

if __name__ == '__main__':
    filepath = 'txt/MANUAL AUDITORIAS_ESP (2).txt'
    
    try:
        file_processor = FileProcessor(filepath)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    original_paragraphs, preprocessed_paragraphs = file_processor.read_and_preprocess()
    model_manager = ModelManager(original_paragraphs, preprocessed_paragraphs)

    while True:
        query = input("Introduce tu consulta (escribe 'salir' para terminar): ")
        if query.lower() == 'salir':
            break

        processed_query = ' '.join(file_processor._clean_and_tokenize(query))

        # Para Word2Vec
        print("Word2Vec:")
        similarities_w2v = model_manager.find_most_similar_sentences_word2vec(processed_query)
        for i, (similarity, _) in enumerate(similarities_w2v, start=1):
            if similarity > 0:
                print(f"El {i}º párrafo más similar a '{query}' es:\n'{original_paragraphs[i-1]}'\nCon una similitud de {similarity}\n")

        # Para TF-IDF
        print("TF-IDF:")
        similarities_tfidf = model_manager.find_most_similar_sentences_tfidf(processed_query)
        for i, (similarity, paragraph) in enumerate(similarities_tfidf, start=1):
            print(f"El {i}º párrafo más similar a '{query}' es:\n'{paragraph}'\nCon una similitud de {similarity}\n")
            
        # Para Word2Vec + TF-IDF
        print("Word2Vec + TF-IDF:")
        similarities_w2v_tfidf = model_manager.find_most_similar_sentences_word2vec_tfidf(query)
        for i, (similarity, paragraph) in enumerate(similarities_w2v_tfidf, start=1):
            if similarity > 0:
                print(f"El {i}º párrafo más similar a '{query}' es:\n'{paragraph}'\nCon una similitud de {similarity}\n")
        

        print("-----------------------------------------------------")