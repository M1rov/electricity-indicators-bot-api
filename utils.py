import nltk

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


nltk.download('punkt')
nltk.download('wordnet')


stemmer = PorterStemmer()

# Ініціалізація лематизатора
lemmatizer = nltk.WordNetLemmatizer()

# Ініціалізація векторайзера
vectorizer = CountVectorizer(max_features=100)

# Ініціалізація кодувальника міток
label_encoder = LabelEncoder()


def tokenize(text):
    """
    splits text into words
    """
    return nltk.word_tokenize(text)


def lemmatize_token(token):
    """
    lemmatizes tokens into lemmatized pattern
    """
    return lemmatizer.lemmatize(token.lower())


def vectorize(words):
    """
    vectorizes text into numeric
    """
    return vectorizer.fit_transform(words).toarray()


def encode_tags(tags):
    """
    encodes tags into numeric
    """
    return label_encoder.fit_transform(tags)

