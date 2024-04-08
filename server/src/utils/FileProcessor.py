import os
from src.utils.TextUtils import TextUtils

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
