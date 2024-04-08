from src.models.Model import Model
from src.utils.TextUtils import TextUtils
import numpy as np
from gensim.models import Word2Vec

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
