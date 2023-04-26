import tweepy
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Twitter API keys and access tokens (Replace with your own keys)
consumer_key = 'aUuTOXF4xuQaxHPoTj1SXs15y'
consumer_secret = 'VPWbSa7QtH3y1pFDiNruhzKZVwyFFZEEVIJx3IcyheFPpArvPU'
access_token = '943525676186869760-S9yLEmxBn6vZraC6boeSi258UJsRHtj'
access_token_secret = 'uOF58rEbsvV8AhKDUryl35IVJpZXxJwbWbvhRGhV7jTTR'


# Function to set up Twitter API
def create_api():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


# Function to clean tweets
def clean_tweet(tweet):
    # Remove mentions, URLs, and non-alphanumeric characters
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


# Function to fetch tweets using Twitter API
def get_tweets(api, query, count=10000):
    tweets = []

    # Get the datetime for 5 hours ago
    since_datetime = datetime.now() - timedelta(hours=6)
    since_str = since_datetime.strftime("%Y-%m-%d")

    # Find a tweet from around 5 hours ago
    tweets_around_five_hours_ago = api.search_tweets(q=query, count=1, lang="en", tweet_mode="extended",
                                                     until=since_str)
    if tweets_around_five_hours_ago:
        since_id = tweets_around_five_hours_ago[0].id
    else:
        since_id = None

    # Use the since_id parameter to fetch tweets from the past 5 hours
    fetched_tweets = api.search_tweets(q=query, count=count, lang="en", tweet_mode="extended", since_id=since_id)

    # Clean and store fetched tweets
    for tweet in fetched_tweets:
        parsed_tweet = {'text': clean_tweet(tweet.full_text)}
        tweets.append(parsed_tweet)

    return tweets


# Fetch and label tweets for training data
def get_labeled_tweets(api, query, count=10000):
    # Fetch tweets using the existing get_tweets function
    tweets = get_tweets(api, query, count)

    # Preprocess and clean tweets
    cleaned_tweets = [clean_tweet(tweet['text']) for tweet in tweets]

    # Label the cleaned tweets (e.g., using a pre-trained sentiment analysis model)
    # In this example, I will use TextBlob, a simple rule-based sentiment analysis library
    from textblob import TextBlob

    def label_sentiment(text):
        sentiment_score = TextBlob(text).sentiment.polarity
        if sentiment_score < -0.05:
            return 0  # negative
        elif sentiment_score > 0.05:
            return 2  # positive
        else:
            return 1  # neutral

    labeled_tweets = [(tweet, label_sentiment(tweet)) for tweet in cleaned_tweets]
    return labeled_tweets


# Function to load a pre-trained LSTM model
def load_pretrained_lstm_model(model_path):
    model = load_model(model_path)
    return model


# Function to create and train a custom LSTM model
def create_and_train_model(train_data, train_labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    max_length = max([len(s.split()) for s in train_data])
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(train_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(padded_sequences, train_labels, epochs=5, batch_size=32)

    return model, tokenizer, max_length


# Function to prepare and predict the sentiment using the custom LSTM model
def predict_sentiment(model, tokenizer, max_length, text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    return np.argmax(prediction, axis=1)[0]


# Function to perform sentiment analysis
def sentiment_analysis(query):
    # Load the pre-trained LSTM model and its corresponding tokenizer and max_length
    model = load_pretrained_lstm_model('pretrained_lstm_model.h5')
    tokenizer = Tokenizer()  # Load the tokenizer you used to train the model
    max_length = 50  # Set the max_length to the value used during training
    api = create_api()
    tweets = get_tweets(api, query)

    tweet_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}

    for tweet in tweets:
        sentiment = predict_sentiment(model, tokenizer, max_length, tweet['text'])

        if sentiment == 0:
            tweet_sentiments['negative'] += 1
        elif sentiment == 1:
            tweet_sentiments['neutral'] += 1
        elif sentiment == 2:
            tweet_sentiments['positive'] += 1

    # Calculate sentiment percentages
    tweet_sentiments_percentages = {k: v / len(tweets) * 100 for k, v in tweet_sentiments.items()}

    # Print sentiment percentages
    print("Sentiment analysis for", query)
    for sentiment, percentage in tweet_sentiments_percentages.items():
        print(f"{sentiment.capitalize()}: {percentage:.2f}%")

    # Create a pie chart to visualize sentiment percentages
    plt.pie(tweet_sentiments_percentages.values(), labels=tweet_sentiments_percentages.keys(), autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title(f"Sentiment analysis for {query}")
    plt.show()


# Main function to take user input and call the sentiment_analysis function
if __name__ == "__main__":
    query = input("Enter the stock or crypto symbol: ")
    # Fetch and label tweets for training data
    api = create_api()
    labeled_tweets = get_labeled_tweets(api, query)
    train_data, train_labels = zip(*labeled_tweets)

    # Train the LSTM model using the labeled tweets
    model, tokenizer, max_length = create_and_train_model(train_data, train_labels)

    # Perform sentiment analysis using the trained model
    sentiment_analysis(query)

    print("Note: This sentiment analysis might not accurately "
          "capture sarcasm or nuanced expressions of sentiment.")
