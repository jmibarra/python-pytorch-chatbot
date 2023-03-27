import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
import numpy as np

# Solo la primera vez / Paquete con un tokenizer para entrenar
nltk.download('punkt')

toktok = ToktokTokenizer()

stemmer = SnowballStemmer('spanish')


def tokenize(sentense):
    return toktok.tokenize(sentense)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    print(tokenized_sentence)

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
