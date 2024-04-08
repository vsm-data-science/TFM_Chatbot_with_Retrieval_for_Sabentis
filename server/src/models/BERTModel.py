from src.models.Model import Model
from transformers import BertTokenizer, BertModel
import torch

class BERTModel(Model):
    def __init__(self, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, [], language)
        self.model, self.tokenizer = self._load_bert_model()

    def _load_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        return model, tokenizer

    def predict(self, query, similarity_threshold):
        query_embedding = self._get_bert_embedding(query)
        similarities = [(self._cosine_similarity(query_embedding, self._get_bert_embedding(paragraph)), paragraph) for paragraph in self.original_paragraphs]
        filtered_and_sorted = sorted([sim for sim in similarities if sim[0] > similarity_threshold], key=lambda x: x[0], reverse=True)
        return filtered_and_sorted

    def _get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
