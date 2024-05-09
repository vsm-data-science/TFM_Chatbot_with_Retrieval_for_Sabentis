from src.models.Model import Model
from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np

class BERTModel(Model):
    def __init__(self, original_paragraphs, language='spanish'):
        super().__init__(original_paragraphs, [], language)
        self.model, self.tokenizer = self._load_bert_model()
        self.embeddings_path = 'embeddings'  # Define the directory to save embeddings

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
        # Ensure the embeddings directory exists
        os.makedirs(self.embeddings_path, exist_ok=True)
        # Generate a file path based on text hash (simple way to avoid file name conflicts)
        hash_id = hash(text)
        file_path = os.path.join(self.embeddings_path, f"{hash_id}.npy")

        if os.path.exists(file_path):
            embeddings = np.load(file_path)
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            np.save(file_path, embeddings)  # Save embeddings to disk

        return embeddings
