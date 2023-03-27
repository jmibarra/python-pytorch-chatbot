import nltk
from nltk.stem import SnowballStemmer
# Solo la primera vez / Paquete con un tokenizer para entrenar
nltk.download('punkt')

stemmer = SnowballStemmer('spanish')


def tokenize(sentense):
    return nltk.word_tokenize(sentense)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass


a = "Hola, ¿en qué puedo ayudarte?"
print(a)
a = tokenize(a)
print(a)

# Tengo que investigar por que al parecer no funciona con palabras en español
words = ['organizar', 'organización', 'organizado']
stemmed_words = [stem(word)for word in words]
print(stemmed_words)
