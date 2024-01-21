import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

lemmatizer = nltk.WordNetLemmatizer()


def tokenize(sentence):
    """
    розділяємо речення на список токенів
    (токеном може бути слово, знак пунктуації, або число)
    """
    return nltk.word_tokenize(sentence)


def lemmatize_token(token):
    """
    лематизуємо слово, тобто беремо його початкову форму
    """
    return lemmatizer.lemmatize(token.lower())


def vectorize(tokenized_sentence, words):
    """
    Перетворює речення у вектор (мішок слів), де кожен елемент вектора відповідає
    присутності слова з заданого словника в реченні. Для кожного слова зі словника,
    вектор містить 1, якщо це слово присутнє у реченні, та 0, якщо відсутнє.
    Приклад:
    tokenized_sentence = ["привіт", "мене", "звуть", "електробот"]
    words = ["хай", "привіт", "як", "справи", "мене", "називають", "звуть"]
    vector   = [0 , 1 , 0 , 0 , 1 , 0 , 1]
    """
    sentence_words = [lemmatize_token(word) for word in tokenized_sentence]
    # ініціалізуємо мультимасив з нулями за довжиною слов
    vector = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            vector[idx] = 1

    return vector
