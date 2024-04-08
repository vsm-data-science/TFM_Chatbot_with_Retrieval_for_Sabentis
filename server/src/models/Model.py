import numpy as np
from scipy import spatial

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
