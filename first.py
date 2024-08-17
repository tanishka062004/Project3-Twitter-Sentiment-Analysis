import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
# Plot libraries
import string
from nltk.stem import PorterStemmer
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Load the data
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")

# Assign column names
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']

# Strip any extra spaces from column names
#df.columns = df.columns.str.strip()

# Print the column names to verify
#print(df.columns)

# View the first five rows of 'sentiment' and 'text'
#print(df[['sentiment', 'text']].head())

# Show shape of the dataset
# print(df.shape)

# #print value counts
# print(df['sentiment'].value_counts())

# # Number of Duplicates
# print(df.duplicated().sum())

# # Drop duplicates
# df.drop_duplicates(inplace=True)

# # Number of Duplicates
# print(df.duplicated().sum())

# Data Exploration Analysis
# df['sentiment'].value_counts().plot(kind='bar', title='Distribution of data', color=['blue'])
# plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
# plt.show()

# Convert date to datetime format
# df['date'] = pd.to_datetime(df['date'])

# # Aggregate the data by date
# df_daily = df.groupby(df['date'].dt.date).count()

# Plot the line graph
# plt.figure(figsize=(10, 6))
# plt.plot(df_daily.index, df_daily['text'], label='Tweet Count')
# plt.title('Number of Tweets over Time')
# plt.xlabel('Date')
# plt.ylabel('Number of Tweets')
# plt.legend()
# plt.grid(True)
# plt.show()


# sentiment_mapping = {0: 'Negative', 4: 'Positive'}
# df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# sentiment_counts = df['sentiment'].value_counts()

# # Custom labels for the x-axis
# x_labels = ['Positive', 'Negative','Neutral']

# # Reindexing based on the order of x_labels
# sentiment_counts = sentiment_counts.reindex(x_labels)


# # # Plotting the distribution
# plt.figure(figsize=(8, 6))
# sns.barplot(x=x_labels, y=sentiment_counts.values, palette='viridis')
# plt.title('Sentiment Distribution')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.show()


#5
# Preprocess the text
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation and non-alphanumeric characters
#     text = re.sub(r'[^\w\s]', '', text)
#     # Tokenize by splitting on whitespace
#     words = text.split()
#     return words

# # Apply preprocessing to the 'text' column
# df['processed_text'] = df['text'].apply(preprocess_text)

# # Flatten the list of lists into a single list of words
# all_words = [word for words in df['processed_text'] for word in words]

# # Count the frequency of each word
# word_freq = Counter(all_words)

# # Generate a word cloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# # Display the word cloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# # Plot the top 20 most common words
# common_words = word_freq.most_common(20)
# words, frequencies = zip(*common_words)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=frequencies, y=words)
# plt.title('Top 20 Most Common Words')
# plt.xlabel('Frequency')
# plt.ylabel('Words')
# plt.show()

#6

df['date'] = pd.to_datetime(df['date'], format='%a %b %d %H:%M:%S PDT %Y', errors='coerce')
df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 4: 'Positive'})

# Set 'date' as the index
df.set_index('date', inplace=True)

# Aggregate sentiment counts by day
daily_sentiment_counts = df.groupby('sentiment_label').resample('D').size().unstack(fill_value=0)

# Reset index to ensure 'date' is a column again
daily_sentiment_counts = daily_sentiment_counts.reset_index()

# Print column names to verify structure
print("Columns in daily_sentiment_counts:", daily_sentiment_counts.columns)

# Convert 'date' column to datetime if needed
if 'date' in daily_sentiment_counts.columns:
    daily_sentiment_counts['date'] = pd.to_datetime(daily_sentiment_counts['date'])
else:
    print("'date' column is missing from the DataFrame.")

# Plot sentiment trends over time
plt.figure(figsize=(14, 7))

# Ensure 'Positive' and 'Negative' columns exist
if 'Positive' in daily_sentiment_counts.columns:
    sns.lineplot(data=daily_sentiment_counts, x='date', y='Positive', label='Positive', color='green')
else:
    print("'Positive' column is missing from the DataFrame.")

if 'Negative' in daily_sentiment_counts.columns:
    sns.lineplot(data=daily_sentiment_counts, x='date', y='Negative', label='Negative', color='red')
else:
    print("'Negative' column is missing from the DataFrame.")

plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.legend()
plt.grid(True)
plt.show()

#Text Processing
# Unique values in column 'sentiment'
print(df['sentiment'].unique())

# Map 0-negative, 4-positive
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})
print(df)

# Removing URLs, Mentions and Websites
def remove_url_mention(text):
    # Define regex patterns
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    site_pattern = r'\b\w+\.com\b'

    # Remove URLs, mentions, websites
    text = re.sub(url_pattern, '', text)  
    text = re.sub(mention_pattern, '', text) 
    text = re.sub(site_pattern, '', text)
    
    return text.strip()
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Remove stopwords
df['wo_stop'] = df['wo_punc'].apply(remove_stopwords)

# Apply Stemming
df['cleaned_text'] = df['wo_stop'].apply(lambda text: stem_words(text))


df['wo_url'] = df['text'].apply(lambda text: remove_url_mention(text))

# Remove Punctuations
translator = str.maketrans('', '', string.punctuation)

df['wo_punc'] = df['wo_url'].apply(lambda text: text.translate(translator))

# Apply Stemming
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["cleaned_text"] = df["wo_stop"].apply(lambda text: stem_words(text))
print(df)


#8
df_sampled = df.sample(frac=0.2, random_state=42)
X = df_sampled['cleaned_text']
y = df_sampled['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vocab_size = len(vectorizer.vocabulary_)
print(f'Vocabulary Size: {vocab_size}')

# Model training
model = RandomForestClassifier(n_estimators=20)
history = model.fit(X_train_vec, y_train)

# Model evaluation
y_pred = model.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Split dataset and vectorize features
X_nb = df['cleaned_text']
y_nb = df['sentiment']

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)

vectorizer_nb = TfidfVectorizer()
X_train_vec_nb = vectorizer.fit_transform(X_train_nb)
X_test_vec_nb = vectorizer.transform(X_test_nb)

# Model Training(2nd Model)
model = MultinomialNB()
model.fit(X_train_vec_nb, y_train_nb)

# Model evaluation
y_pred_nb = model.predict(X_test_vec_nb)
print(f'Accuracy: {accuracy_score(y_test_nb, y_pred_nb)}')
print(classification_report(y_test_nb, y_pred_nb))

cm = confusion_matrix(y_test_nb, y_pred_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
