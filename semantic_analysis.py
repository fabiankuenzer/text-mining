import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

nltk.download('stopwords')


def truncated_svd_document_topic_relevance(vectors, number_of_topics):
    lsa_model = TruncatedSVD(n_components=number_of_topics, algorithm='randomized')
    document_topic_relevance = lsa_model.fit_transform(vectors)
    topics = lsa_model.components_
    return document_topic_relevance, topics


def extract_topics_and_most_important_keywords(topics, feature_names, n_top_words):
    for index, topic in enumerate(topics):
        print(f"Topic {index + 1}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print('')


def latent_dirichlet_allocation_document_topic_relevance(vectors, number_of_topics):
    lda_model = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online')
    document_topic_relevance = lda_model.fit_transform(vectors)
    topics = lda_model.components_
    return document_topic_relevance, topics


def get_topics_and_relevance(document_topic_relevance):
    for document_index, document in enumerate(document_topic_relevance):
        for topic_index, topic_relevance in enumerate(document):
            print("Topic ", topic_index, ": ", topic_relevance * 100, "%")
        print('')


def get_topic_distribution(document_topic_relevance, n_topics):
    most_important_topics = []
    topic_distribution = []
    for document in document_topic_relevance:
        most_important_topics.append(np.where(document == np.max(document))[0])
    for count in range(0, n_topics):
        topic_distribution.append(most_important_topics.count(count))
    return topic_distribution


def get_stopwords_excluding(list_of_stopwords_to_exclude):
    stopword_list = stopwords.words('english')
    print(type(stopword_list))
    for word in list_of_stopwords_to_exclude:
        stopword_list.remove(word)
    return stopword_list
