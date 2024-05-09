from nltk.corpus import stopwords
from src.models.Model import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle

class TFIDFModel(Model):
    def __init__(self, preprocessed_paragraphs, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, preprocessed_paragraphs, language)
        self.model, self.feature_names = self._train_model()

    def _train_model(self):
        vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopwords.words(self.language))
        tfidf_matrix = vectorizer.fit_transform(self.preprocessed_paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        # Save the TF-IDF matrix and vectorizer
        np.save('tfidf_matrix.npy', tfidf_matrix.toarray())  # Save the matrix
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)  # Save the vectorizer state
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
