import os
from src.models.Langchain import LangChain
from src.models.ModelManager import ModelManager
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from enum import Enum

class ModelTypes(Enum):
    legacy = 'legacy'
    langchain = 'langchain'

langchain_model = LangChain()

def download_nltk_packages():
    packages = ['stopwords', 'punkt']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_packages()

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']
    model = data['model'] if 'model' in data else ModelTypes.langchain.value

    if model == ModelTypes.legacy.value:
        file, results = ModelManager.find_most_similar_dictionary(query)
        chat_gpt_answer = ModelManager.get_chat_gpt_answer(file, results, query)
        
        return jsonify({
            'chat_gpt_answer': chat_gpt_answer,
            'file': file,
            'results': results
        })
    else:
        return jsonify({
            'chat_gpt_answer': langchain_model.predict(query)
        })

if __name__ == '__main__':
    app.run()
