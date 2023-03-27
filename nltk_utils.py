import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer

# Solo la primera vez / Paquete con un tokenizer para entrenar
nltk.download('punkt')

toktok = ToktokTokenizer()

stemmer = SnowballStemmer('spanish')


def tokenize(sentense):
    return toktok.tokenize(sentense)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass
