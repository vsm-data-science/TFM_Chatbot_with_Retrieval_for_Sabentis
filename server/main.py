import os
from src.models.ModelManager import ModelManager
import nltk
from flask import Flask, request, jsonify


def download_nltk_packages():
    packages = ['stopwords', 'punkt']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_packages()

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']
    
    file, results = ModelManager.find_most_similar_dictionary(query)
    chat_gpt_answer = ModelManager.get_chat_gpt_answer(file, results, query)
    
    return jsonify({
        'chat_gpt_answer': chat_gpt_answer,
        'file': file,
        'results': results
    })

if __name__ == '__main__':
    app.run()
