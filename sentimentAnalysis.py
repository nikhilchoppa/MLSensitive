import praw
import os
import re
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping


# Reddit API keys and access tokens
client_id = 'GGIKF1inKH3_M-fidHcHoA'
client_secret = '-XM0G4dX6eKi-ldvt4-a0hT7bPzQEQ'
user_agent = 'ml2'
username = 'Dazzling-Resident988'
password = 'Nikhil009'


# Function to set up Reddit API
def create_api():
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password
    )
    return reddit


def clean_title(title):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', title).split())


def get_titles(api, query, count=1000):
    titles = []
    subreddit = api.subreddit(query)

    for submission in subreddit.hot(limit=count):
        parsed_title = {'text': clean_title(submission.title)}
        titles.append(parsed_title)

    return titles


def get_labeled_titles(api, query, count=1000):
    titles = get_titles(api, query, count)
    cleaned_titles = [clean_title(title['text']) for title in titles]

    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    labeled_titles = []
    for title in cleaned_titles:
        sentiment = sia.polarity_scores(title)['compound']
        if sentiment >= 0.05:
            labeled_titles.append((title, 2))
        elif sentiment <= -0.05:
            labeled_titles.append((title, 0))
        else:
            labeled_titles.append((title, 1))

    return labeled_titles


def create_model(tokenizer, labels_count):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=40),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(labels_count, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


# Changed function names and variables to match Reddit-specific functions
def train_and_save_model(query):
    api = create_api()
    labeled_titles = get_labeled_titles(api, query)
    train_data = [title for title, label in labeled_titles]
    train_labels = [label for title, label in labeled_titles]

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data)

    train_sequences = tokenizer.texts_to_sequences(train_data)
    padded_train_sequences = pad_sequences(train_sequences, maxlen=40, truncating="post", padding="post")

    X_train, X_test, y_train, y_test = train_test_split(padded_train_sequences, np.array(train_labels), test_size=0.33,
                                                        random_state=42)

    model = create_model(tokenizer, labels_count=3)

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, callbacks=[early_stopping])

    model_file = f'{query}_model.h5'
    tokenizer_file = f'{query}_tokenizer.pickle'

    model.save(model_file)

    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Changed function names and variables to match Reddit-specific functions

def predict_sentiment(query, text):
    model_file = f'{query}_model.h5'
    tokenizer_file = f'{query}_tokenizer.pickle'
    if not os.path.exists(model_file) or not os.path.exists(tokenizer_file):
        train_and_save_model(query)

    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model(model_file)

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=40, truncating="post", padding="post")

    return model.predict(padded_sequence)


query = input("Enter the subreddit name: ").lower()
titles = get_titles(create_api(), query, count=100)
sentiments = {"positive": 0, "neutral": 0, "negative": 0}

for title in titles:
    text = clean_title(title['text'])
    sentiment_prediction = predict_sentiment(query, text)
    sentiment_label = sentiment_prediction.argmax(axis=-1)[0]
    if sentiment_label == 0:
        sentiments["negative"] += 1
    elif sentiment_label == 1:
        sentiments["neutral"] += 1
    else:
        sentiments["positive"] += 1

total = sum(sentiments.values())

positive_percentage = sentiments["positive"] / total * 100
negative_percentage = sentiments["negative"] / total * 100
neutral_percentage = sentiments["neutral"] / total * 100

print(f'Sentiment analysis for r/{query}')
print(f'Positive: {positive_percentage:.2f}%')
print(f'Negative: {negative_percentage:.2f}%')
print(f'Neutral: {neutral_percentage:.2f}%')

# Create the pie chart
sizes = [positive_percentage, negative_percentage, neutral_percentage]
labels = ['Positive', 'Negative', 'Neutral']
colors = ['#33FF90', '#FF4500', '#B0C4DE']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')

# Display the chart
plt.show()

print('\nNote: This sentiment analysis might not accurately capture sarcasm or nuanced expressions of sentiment.')
