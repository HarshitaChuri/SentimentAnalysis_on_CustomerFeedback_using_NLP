import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# No need to download NLTK resources if already installed
# nltk.download('stopwords')

# Function to categorize sentiment scores
def categorize_sentiment(score):
    if score > 0.8:
        return "Highly Positive"
    elif score > 0.4:
        return "Positive"
    elif -0.4 <= score <= 0.4:
        return "Neutral"
    elif score < -0.4:
        return "Negative"
    else:
        return "Highly Negative"

# Function to clean text with NLTK
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters (retain spaces for stopwords)
    text = text.lower()  # Convert to lowercase
    words = text.split()
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Load the data
df = pd.read_csv("reviews.csv")
review_text = df['Text']

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment scores and subjectivity
sentiment_scores = []
blob_subj = []
for review in review_text:
    cleaned_review = clean_text(review)
    sentiment_scores.append(analyzer.polarity_scores(cleaned_review)["compound"])
    blob = TextBlob(cleaned_review)
    blob_subj.append(blob.subjectivity)

sentiment_classes = [categorize_sentiment(score) for score in sentiment_scores]

# Streamlit UI
st.title("Sentiment Analysis on Customer Feedback")
st.markdown("---")

# Main content area
st.markdown("### Graphical Representation of Data")
plt.figure(figsize=(10,6))
sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, bins=20, label=sentiment_class, alpha=0.5)

plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.title("Score Distribution by Sentiment Class")
plt.grid(True)
plt.legend()
st.pyplot(plt)

# Display DataFrame
st.markdown("### Input Dataframe")
st.dataframe(df.head(10), height=300)

# Text Cleaning Section
st.markdown("### Clean Text Example")
user_input = st.text_area("Enter Text to Clean:", height=100)
if st.button("Clean Text", key="clean_button"):
    cleaned_text = clean_text(user_input)
    st.write("**Original Text:**", user_input)
    st.write("**Cleaned Text:**", cleaned_text)

# Sentiment Analysis Section
st.markdown("### Sentiment Analysis Example")
sentiment_example = st.selectbox("Select an example review:", df['Text'].values)
cleaned_example = clean_text(sentiment_example)
analyzer_score = analyzer.polarity_scores(cleaned_example)
blob = TextBlob(cleaned_example)

st.write("**Selected Text:**", sentiment_example)
st.write("**Cleaned Text:**", cleaned_example)
st.write("**VADER Sentiment Score:**", analyzer_score['compound'])
st.write("**VADER Sentiment Class:**", categorize_sentiment(analyzer_score['compound']))
st.write("**TextBlob Polarity:**", blob.sentiment.polarity)
st.write("**TextBlob Subjectivity:**", blob.sentiment.subjectivity)

# Footer
st.markdown("---")
st.markdown("Developed by Harshita Churi !")
