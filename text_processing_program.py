import os
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy import spatial
import unidecode

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Función para limpiar y tokenizar el texto
def clean_and_tokenize(text, language='spanish'):
    text = unidecode.unidecode(text)
    stop_words = set(stopwords.words(language))
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text, language=language)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# Función para entrenar el modelo Word2Vec
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Función para obtener el embedding de una oración utilizando el modelo Word2Vec
def get_sentence_embedding(sentence, word2vec_model, language='spanish'):
    words = clean_and_tokenize(sentence, language=language)
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) == 0:
        return np.zeros(word2vec_model.vector_size)
    sentence_embedding = np.mean(word_vectors, axis=0)
    return sentence_embedding

# Función para calcular la similitud coseno entre dos embeddings
def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return -1  # Retorna una similitud de -1 si uno de los vectores es todo ceros
    return 1 - spatial.distance.cosine(vec1, vec2)

# Función para encontrar las dos oraciones más similares a una consulta en un conjunto de oraciones
def find_most_similar_sentences(query, paragraphs, word2vec_model, language='spanish'):
    query_embedding = get_sentence_embedding(query, word2vec_model, language=language)
    top_similarities = [(float('-inf'), ""), (float('-inf'), "")]

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_tokens = clean_and_tokenize(paragraph, language=language)  # Tokeniza el párrafo completo
        if not paragraph_tokens:
            continue

        # Calcula el embedding del párrafo completo en lugar de oraciones individuales
        paragraph_embedding = get_sentence_embedding(' '.join(paragraph_tokens), word2vec_model, language=language)
        similarity = cosine_similarity(query_embedding, paragraph_embedding)

        if similarity > top_similarities[0][0]:
            top_similarities[0] = (similarity, paragraph)  # Almacena el párrafo completo
            top_similarities.sort(reverse=True)  # Ordena para que las mejores similitudes estén al principio

    top_similarities = [(max(0, sim), para) for sim, para in top_similarities]  # Asegúrate de que la similitud no sea negativa
    return top_similarities


if __name__ == '__main__':
    # Leer el contenido completo del archivo .txt
    with open('txt/MANUAL AUDITORIAS_ESP (2).txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # Dividir el contenido en párrafos usando dos saltos de línea como delimitador
    paragraphs = content.split('\n\n')

    # Asegúrate de que cada párrafo se maneje correctamente, eliminando espacios adicionales
    paragraphs = [para.strip() for para in paragraphs if para.strip() != '']

    # Preprocesar y tokenizar cada párrafo para el entrenamiento del modelo Word2Vec
    sentences = [clean_and_tokenize(paragraph, language='spanish') for paragraph in paragraphs]

    # Entrenar el modelo Word2Vec con las frases tokenizadas
    word2vec_model = train_word2vec(sentences)

    while True:  # Bucle infinito hasta que el usuario decida salir
        query = input("Introduce tu consulta (escribe 'salir' para terminar): ")  # Solicita la consulta al usuario
        if query.lower() == 'salir':  # Si el usuario escribe 'salir', termina el bucle
            break

        # Encontrar los párrafos que contienen las dos oraciones más similares a la consulta
        similarities = find_most_similar_sentences(query, paragraphs, word2vec_model, language='spanish')

        # Iterar a través de las similitudes e imprimir solo aquellas con similitud mayor que 0
        for i, (similarity, paragraph) in enumerate(similarities, start=1):
            if similarity > 0:  # Verifica que la similitud sea mayor que 0 antes de imprimir
                print(f"El {i}º párrafo más similar a '{query}' es:\n'{paragraph}'\nCon una similitud de {similarity}\n")
            else:
                break  # Si la similitud es 0 o menor, no hay necesidad de seguir iterando

        print("-----------------------------------------------------")  # Imprime una línea divisoria entre consultas
