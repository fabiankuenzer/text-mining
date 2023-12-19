import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from semantic_context import get_semantic_context
from vectorization import *
from semantic_analysis import *
from preprocessing import *
from domain_specific_stopwords import domain_specific_stopwords

nltk.download('stopwords')


print('IMPORTING AND EXTRACTING DATA: START')
raw_data = pd.read_csv('data/comcast.csv')
corpus = list(raw_data['text'].dropna())
print('IMPORTING AND EXTRACTING DATA: SUCCESS')

print('PREPARING DATA FOR ANALYSIS: START')
custom_stopwords = stopwords.words('english')
custom_stopwords.extend(["n't", "'m", "w/", "'d", "'s", "min", "00", "3rd", "4th", "49.99", "19.99", "50.00", "comcast",
                         "mbps", "since", "one", "rep", "would", "really", "bla", "my", "mr", "08"])
custom_stopwords.extend([str(number) for number in range(0, 5000)])
custom_stopwords.extend(domain_specific_stopwords)
filtered_corpus = [remove_stopwords(document, custom_stopwords) for document in corpus]
lowercase_corpus = [lowercase_tokens(document) for document in filtered_corpus]
corpus_without_punctuation = [remove_punctuation(document) for document in lowercase_corpus]
lemmatized_corpus = [lemmatization(document) for document in corpus_without_punctuation]
print('PREPARING DATA FOR ANALYSIS: SUCCESS')
print('')

print('VECTORIZATION: START')
count_vectors, count_feature_names = count_vectorization(lemmatized_corpus, (1, 1))
tfidf_vectors, tfidf_feature_names = tfidf_vectorization(lemmatized_corpus, (1, 1))
print('VECTORIZATION: SUCCESS')
print('')

print('SET ANALYSIS PARAMETERS: START')
number_of_topics_to_extract = int(input('Enter number of topics to extract: '))
number_of_top_words_per_topic = int(input('Enter number of most important words to extract per topic: '))
print('SET ANALYSIS PARAMETERS: SUCCESS')
print('')

print("---LSA with count vectorization---")
print('')
lsa_count_document_topic_relevance, lsa_count_topics = truncated_svd_document_topic_relevance(count_vectors, number_of_topics_to_extract)
extract_topics_and_most_important_keywords(lsa_count_topics, count_feature_names, number_of_top_words_per_topic)
print('Topic distribution:')
lsa_count_number_of_documents_per_topic = get_topic_distribution(lsa_count_document_topic_relevance, number_of_topics_to_extract)
print_share_of_documents_per_topic(lsa_count_number_of_documents_per_topic)
print('')

print("---LSA with TF-IDF vectorization---")
print('')
lsa_tfidf_document_topic_relevance, lsa_tfidf_topics = truncated_svd_document_topic_relevance(tfidf_vectors, number_of_topics_to_extract)
extract_topics_and_most_important_keywords(lsa_tfidf_topics, tfidf_feature_names, number_of_top_words_per_topic)
print('Topic distribution:')
lsa_tfidf_number_of_documents_per_topic = get_topic_distribution(lsa_tfidf_document_topic_relevance, number_of_topics_to_extract)
print_share_of_documents_per_topic(lsa_tfidf_number_of_documents_per_topic)
print('')

print("---LDA with count vectorization---")
print('')
lda_count_document_topic_relevance, lda_count_topics = latent_dirichlet_allocation_document_topic_relevance(count_vectors, number_of_topics_to_extract)
extract_topics_and_most_important_keywords(lda_count_topics, count_feature_names, number_of_top_words_per_topic)
print('Topic distribution:')
lda_count_number_of_documents_per_topic = get_topic_distribution(lda_count_document_topic_relevance, number_of_topics_to_extract)
print_share_of_documents_per_topic(lda_count_number_of_documents_per_topic)
print('')

print("---LDA with TF-IDF vectorization---")
print('')
lda_tfidf_document_topic_relevance, lda_tfidf_topics = latent_dirichlet_allocation_document_topic_relevance(tfidf_vectors, number_of_topics_to_extract)
extract_topics_and_most_important_keywords(lda_tfidf_topics, tfidf_feature_names, number_of_top_words_per_topic)
print('Topic distribution:')
lda_tfidf_number_of_documents_per_topic = get_topic_distribution(lda_tfidf_document_topic_relevance, number_of_topics_to_extract)
print_share_of_documents_per_topic(lda_tfidf_number_of_documents_per_topic)
print('')

print("---Explore words that are used in a similar semantic context as the given topic---")
print("TRAIN MODEL: START")
tokenized_corpus = [word_tokenize(doc) for doc in lemmatized_corpus]
word2vec_model = Word2Vec(tokenized_corpus)
print("TRAIN MODEL: SUCCESS")
get_semantic_context(word2vec_model)
