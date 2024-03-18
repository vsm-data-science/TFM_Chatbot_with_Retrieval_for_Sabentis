import os
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy import spatial
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # Para calcular la similitud coseno con vectores TF-IDF
import nltk

nltk.download('stopwords')
nltk.download('punkt')


class TextSimilarityFinder:
    def __init__(self, filepath, language='spanish'):
        self.language = language
        self.filepath = filepath
        self.paragraphs = self._read_file()
        self.word2vec_model = self._train_word2vec_model()
        self.tfidf_matrix, self.tfidf_feature_names = self._train_tfidf_model()
    
    def _read_file(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        paragraphs = content.split('\n\n')
        return [para.strip() for para in paragraphs if para.strip() != '']
    
    def _clean_and_tokenize(self, text):
        text = unidecode.unidecode(text)
        stop_words = set(stopwords.words(self.language))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        tokens = word_tokenize(text, language=self.language)
        return [token for token in tokens if token not in stop_words]
    
    def _train_word2vec_model(self):
        sentences = [self._clean_and_tokenize(paragraph) for paragraph in self.paragraphs]
        return Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    def _get_sentence_embedding(self, sentence):
        words = self._clean_and_tokenize(sentence)
        word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if len(word_vectors) == 0:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(word_vectors, axis=0)
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return -1
        return 1 - spatial.distance.cosine(vec1, vec2)
    
    def _train_tfidf_model(self):
        # Unimos los párrafos para el modelo TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language))
        tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(self._clean_and_tokenize(para)) for para in self.paragraphs])
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        return tfidf_matrix, tfidf_feature_names
    
    def find_most_similar_sentences_tfidf(self, query):
        query_vec = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language), vocabulary=self.tfidf_feature_names).fit_transform([' '.join(self._clean_and_tokenize(query))])
        cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_similarities_indices = cosine_similarities.argsort()[:-3:-1]  # Obtiene los índices de las dos mejores similitudes
        return [(cosine_similarities[i], self.paragraphs[i]) for i in top_similarities_indices]

    
    def find_most_similar_sentences(self, query):
        query_embedding = self._get_sentence_embedding(query)
        top_similarities = [(float('-inf'), ""), (float('-inf'), "")]

        for paragraph in self.paragraphs:
            paragraph_tokens = self._clean_and_tokenize(paragraph)
            if not paragraph_tokens:
                continue

            paragraph_embedding = self._get_sentence_embedding(' '.join(paragraph_tokens))
            similarity = self._cosine_similarity(query_embedding, paragraph_embedding)

            if similarity > top_similarities[0][0]:
                top_similarities[0] = (similarity, paragraph)
                top_similarities.sort(reverse=True)

        return [(max(0, sim), para) for sim, para in top_similarities]


if __name__ == '__main__':
    similarity_finder = TextSimilarityFinder('txt/MANUAL AUDITORIAS_ESP (2).txt')

    while True:
        query = input("Introduce tu consulta (escribe 'salir' para terminar): ")
        if query.lower() == 'salir':
            break

        # Para Word2Vec
        print("Word2Vec:")
        similarities_w2v = similarity_finder.find_most_similar_sentences(query)
        for i, (similarity, paragraph) in enumerate(similarities_w2v, start=1):
            if similarity > 0:
                print(f"El {i}º párrafo más similar a '{query}' es:\n'{paragraph}'\nCon una similitud de {similarity}\n")

        # Para TF-IDF
        print("TF-IDF:")
        similarities_tfidf = similarity_finder.find_most_similar_sentences_tfidf(query)
        for i, (similarity, paragraph) in enumerate(similarities_tfidf, start=1):
            print(f"El {i}º párrafo más similar a '{query}' es:\n'{paragraph}'\nCon una similitud de {similarity}\n")

        print("-----------------------------------------------------")
