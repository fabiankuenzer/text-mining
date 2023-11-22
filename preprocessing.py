import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer

nltk.download('wordnet')
nltk.download('punkt')


def remove_stopwords(document, stopwords):
    words = word_tokenize(str(document))
    filtered_words = [word for word in words if word.lower() not in stopwords]
    filtered_document = ' '.join(filtered_words)
    return filtered_document


def lowercase_tokens(document):
    words = word_tokenize(str(document))
    lowercase_words = [word.lower() for word in words]
    lowercase_document = ' '.join(lowercase_words)
    return lowercase_document


def remove_punctuation(document):
    words = word_tokenize(str(document))
    words_without_punctuation = [word for word in words if word not in list(string.punctuation)]
    document_without_punctuation = ' '.join(words_without_punctuation)
    return document_without_punctuation


def lemmatization(document):
    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(str(document))
    result = list(map(lambda word: stemmer.stem(word) if word.endswith("ing") else lemmatizer.lemmatize(word), words))
    lemmatized_document = ' '.join(result)
    return lemmatized_document
