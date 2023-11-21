import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

nltk.download('vader_lexicon')


def sentiment_intensity_analysis(text):
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(text))


def transformers_sentiment_analysis(text):
    sent_pipeline = pipeline('sentiment-analysis')
    print(sent_pipeline(text))


review_1 = "Loved the sound, No battery issues"
review_2 = "Sound quality is good; battery life not good"

print(sentiment_intensity_analysis(review_1))
print(sentiment_intensity_analysis(review_2))
