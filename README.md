# MLSensitive

If you look at the last decades, the markets are getting more and more influenced by what’s happening on news and on Twitter. For Instance, if you look at just a few years ago on the 7th of January 2021 after Elon Must have tweeted “Use Signal” which sold by 1000% later on. Like these tweets, there are many more which are highly influenced on the market. That’s why we are going to build messenger needs an assistant in your to smoothen in order to predict what’s going to happen on the market.

## Plotly Library - For Finance and Algorithmic Trading

  - Uses to develop online data analytics and visualization tools.
  - Plotly provides graph analytics and statistics tools through python.
  - Plot complex charts
  - Effectiveness

https://docs.google.com/document/d/1UXvbnfJk0oSzkn_lJ_ez_ZUm7QBzxn_sSHYImB8wlMY/edit?usp=sharing

## Create an Environment

- `conda create -n mlenv python=3.10 anaconda`
- `conda activate mlenv`
- `conda install pandas matplotlib`
- `conda install tensorflow`
- `conda install transformers`
- Install Pytorch according to your System configarations:`https://pytorch.org/get-started/locally/`
- Now your environment is somplete settled
- Add this Environment to the Jupyter Notebook
- To do that you need to install ipykernel `python -m ipykernel install --user --name mlenv --display-name "ML environment"`


## About SentimentalAnalysis.py

This code performs sentiment analysis on tweets containing a given stock or crypto symbol using the Twitter API, the Hugging Face Transformers library, and a pre-trained DistilBERT model. Here's a step-by-step breakdown of the code:

1. Import the required libraries, including Tweepy (for Twitter API access), regex (for tweet cleaning), Pandas, NumPy, and Matplotlib (for data manipulation and visualization), Transformers (for the DistilBERT model), and TensorFlow.

2. Define the Twitter API keys and access tokens.

3. Define the `create_api()` function to set up and authenticate the Twitter API.

4. Define the `clean_tweet()` function to remove mentions, URLs, and non-alphanumeric characters from the tweet text.

5. Define the get_tweets() function to fetch tweets using the Twitter API:

    - Determine the datetime for 5 hours ago.
    - Find a tweet from around 5 hours ago. 
    - Fetch tweets from the past 5 hours containing the query (stock or crypto symbol). 
    - Clean and store the fetched tweets.

6. Define the load_model() function to load the pre-trained DistilBERT model and tokenizer for sentiment analysis.

7. Define the sentiment_analysis() function:

    - Set up the Twitter API and fetch tweets using the query.
    - Load the DistilBERT model and tokenizer.
    - Initialize tweet sentiment counters (positive and negative).
    - Classify the sentiment of each tweet using the model.
    - Calculate sentiment percentages.
    - Print sentiment percentages.
    - Create a pie chart to visualize sentiment percentages.

8. Define the main function to take user input (stock or crypto symbol) and call the sentiment_analysis() function. A note is also printed, reminding users that the sentiment analysis might not accurately capture sarcasm or nuanced expressions of sentiment.

When executed, the code takes a stock or crypto symbol as input, fetches tweets containing that symbol from the past 5 hours, performs sentiment analysis using the DistilBERT model, and displays the sentiment percentages in the console and as a pie chart.
