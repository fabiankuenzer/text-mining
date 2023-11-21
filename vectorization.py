from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tfidf_vectorization(corpus, n_gram_range):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', lowercase=True, ngram_range=n_gram_range)
    vectors = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return vectors, feature_names


def count_vectorization(corpus, n_gram_range):
    count_vectorizer = CountVectorizer(stop_words='english', lowercase=True, ngram_range=n_gram_range)
    vectors = count_vectorizer.fit_transform(corpus)
    feature_names = count_vectorizer.get_feature_names_out()
    return vectors, feature_names
