from src.utils.TextUtils import TextUtils
from src.models.TFIDFModel import TFIDFModel
from src.models.Word2VecModel import Word2VecModel
from nltk.corpus import stopwords
from src.models.Model import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
