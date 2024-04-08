import os
from src.utils import TextUtils
from src.models.Model import Model
from gensim.models import KeyedVectors
import numpy as np

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