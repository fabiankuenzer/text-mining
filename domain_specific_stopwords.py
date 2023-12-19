from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


raw_data = pd.read_csv('data/comcast.csv')
corpus = list(raw_data['text'].dropna())


def find_n_most_common_words(corpus, n):
    vocabulary = " ".join(corpus)
    words = word_tokenize(vocabulary)
    lowercase_words = [word.lower() for word in words]
    count_dict = dict(Counter(lowercase_words))
    sorted_keys = sorted(count_dict, key=count_dict.get, reverse=True)
    return sorted_keys[:n]


most_common_words = find_n_most_common_words(corpus, 250)
potential_domain_specific_stopwords = list(set(most_common_words) - set(stopwords.words('english')))

domain_specific_stopwords = ['...', 'us', '!', "'s", '(', '$', '*', '.', 'could', '&', '?', 'ca', "'m", 'would', "''",
                            "'ve", "n't", 'rep', ')', '``', ',', '-', 'told', 'call', 'called', 'get', 'said']
