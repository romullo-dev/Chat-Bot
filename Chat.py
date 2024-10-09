import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixar pacotes necessários do NLTK (se ainda não baixou)
nltk.download('punkt')
nltk.download('wordnet')

# Exemplo simples de um corpus de conversas
corpus = """Olá! Como posso ajudar você hoje? 
Eu gostaria de saber mais sobre programação em Python.
Python é uma linguagem de programação poderosa e fácil de aprender.
Ela é muito usada para desenvolvimento web, automação, análise de dados e muito mais.
Você pode me dizer o que são loops em Python?
Um loop é uma estrutura de controle que repete um bloco de código enquanto uma condição for verdadeira.
Existem dois tipos principais de loops em Python: o 'for' e o 'while'.
Posso automatizar tarefas com Python?
Sim, Python é excelente para automação. Você pode automatizar tarefas repetitivas como mover arquivos, renomear arquivos e extrair dados da web.
Obrigado! Isso foi muito útil.
De nada! Estou aqui para ajudar.
Até logo!
Até logo!"""

# Tokenização do corpus
sent_tokens = nltk.sent_tokenize(corpus.lower())

# Lematização (redução de palavras às suas formas básicas)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punctuation)))

# Função de resposta do chatbot
def resposta_do_chatbot(entrada_usuario):
    sent_tokens.append(entrada_usuario)
    tfidf_vec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidf_vec.fit_transform(sent_tokens)
    similaridades = cosine_similarity(tfidf[-1], tfidf)
    indice_similar = similaridades.argsort()[0][-2]
    similaridade_do_texto = similaridades.flatten()
    similaridade_do_texto.sort()
    melhor_resposta = similaridade_do_texto[-2]

    if melhor_resposta == 0:
        resposta = "Desculpe, não entendi isso. Pode repetir?"
    else:
        resposta = sent_tokens[indice_similar]
    
    sent_tokens.pop()  # Remove a entrada do usuário adicionada temporariamente
    return resposta

# Função principal do chatbot
def chatbot():
    print("Chatbot: Olá! Eu sou o chatbot. Pergunte algo e tentarei ajudar. (Digite 'sair' para encerrar)")
    while True:
        entrada_usuario = input("Você: ").lower()
        if entrada_usuario == 'sair':
            print("Chatbot: Até logo! Foi bom conversar com você!")
            break
        else:
            print("Chatbot:", resposta_do_chatbot(entrada_usuario))

# Iniciar o chatbot
chatbot()

