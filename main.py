import pandas as pd
from vectorization import *
from semantic_analysis import *
from preprocessing import *


print('IMPORTING AND EXTRACTING DATA: START')
raw_data = pd.read_csv('data/comcast.csv')
corpus = list(raw_data['text'][0:10])
print('IMPORTING AND EXTRACTING DATA: SUCCESS')

print('PREPARING DATA FOR ANALYSIS: START')
custom_stopwords = stopwords.words('english')
# custom_stopwords = remove_words_from_nltk_stopwords(stopwords, ['does', 'do', 'no', 'not'])
# Todo - add n't 'm to custom_stopwords with respective function
filtered_corpus = [remove_stopwords(document, custom_stopwords) for document in corpus]
lowercase_corpus = [lowercase_tokens(document) for document in filtered_corpus]
corpus_without_punctuation = [remove_punctuation(document) for document in lowercase_corpus]
lemmatized_corpus = [lemmatization(document) for document in corpus_without_punctuation]
print('PREPARING DATA FOR ANALYSIS: SUCCESS')
print('')
print('FINAL CORPUS SAMPLE')
print(lemmatized_corpus[:3])
print('')

print('VECTORIZATION: START')
count_vectors, count_feature_names = count_vectorization(corpus_without_punctuation, (1, 3))
tfidf_vectors, tfidf_feature_names = tfidf_vectorization(corpus_without_punctuation, (1, 3))
print('VECTORIZATION: SUCCESS')
print('')

print("---LSA with count vectorization---")
print('')
lsa_count_document_topic_relevance, lsa_count_topics = truncated_svd_document_topic_relevance(count_vectors, 2)
extract_topics_and_most_important_keywords(lsa_count_topics, count_feature_names, 3)
# get_topics_and_relevance(lsa_count_document_topic_relevance)
print(get_topic_distribution(lsa_count_document_topic_relevance, 2))
print('')

print("---LSA with TF-IDF vectorization---")
print('')
lsa_tfidf_document_topic_relevance, lsa_tfidf_topics = truncated_svd_document_topic_relevance(tfidf_vectors, 2)
extract_topics_and_most_important_keywords(lsa_tfidf_topics, tfidf_feature_names, 3)
# get_topics_and_relevance(lsa_tfidf_document_topic_relevance)
print(get_topic_distribution(lsa_tfidf_document_topic_relevance, 2))
print('')

print("---LDA with count vectorization---")
print('')
lda_count_document_topic_relevance, lda_count_topics = latent_dirichlet_allocation_document_topic_relevance(count_vectors, 2)
extract_topics_and_most_important_keywords(lda_count_topics, count_feature_names, 3)
# get_topics_and_relevance(lda_count_document_topic_relevance)
print(get_topic_distribution(lda_count_document_topic_relevance, 2))
print('')

print("---LDA with TF-IDF vectorization---")
print('')
lda_tfidf_document_topic_relevance, lda_tfidf_topics = latent_dirichlet_allocation_document_topic_relevance(tfidf_vectors, 2)
extract_topics_and_most_important_keywords(lda_tfidf_topics, tfidf_feature_names, 3)
# get_topics_and_relevance(lda_tfidf_document_topic_relevance)
print(get_topic_distribution(lda_tfidf_document_topic_relevance, 2))
print('')
