import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode

class TextUtils:
    @staticmethod
    def clean_and_tokenize(text, language):
        text = unidecode.unidecode(text.lower())
        stop_words = set(stopwords.words(language))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        tokens = word_tokenize(text, language=language)
        return [token for token in tokens if token not in stop_words]
