import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation


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


def print_share_of_documents_per_topic(list_of_documents_per_topic):
    total_document_count = sum(list_of_documents_per_topic)
    for index, document_count in enumerate(list_of_documents_per_topic):
        print("Share of documents assigned to topic", index, ":", round(document_count/total_document_count*100, 2), "%")
