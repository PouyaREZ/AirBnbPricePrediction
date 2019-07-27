from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime as dt
# Importing the dataset
filename = '../Data/reviews_original.csv'
data = pd.read_csv(filename)
#join on id and listing id

data = pd.DataFrame.drop(data, columns=[
    'id',
    'date',
    'reviewer_id',
    'reviewer_name'


])
def calculate_sentiment(entry):
    if (type(entry) != str and math.isnan(entry)):
        return -55
    opinion = TextBlob(entry)
    return opinion.sentiment.polarity


data['comments'] = data['comments'].apply(calculate_sentiment)
data = data[data['comments'] != -55]
data = data.groupby('listing_id')['comments']. mean()
data.to_csv('../Data/reviews_cleaned.csv')
