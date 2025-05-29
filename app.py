
import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# App title
st.title("📊 Customer Review Sentiment Analyzer")

st.write("This app analyzes sentiment from uploaded CSV reviews and manual input.")

# --- Section 1: Upload CSV file ---
st.header("📂 Upload Customer Reviews CSV")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            df = df.dropna(subset=['review'])

            # Analyze sentiment scores
            def analyze_sentiment(text):
                scores = sia.polarity_scores(text)
                compound = scores['compound']
                if compound >= 0.05:
                    sentiment = "Positive"
                elif compound <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                return pd.Series([scores['neg'], scores['neu'], scores['pos'], compound, sentiment],
                                 index=['Negative', 'Neutral', 'Positive', 'Compound', 'Sentiment'])

            df[['Negative', 'Neutral', 'Positive', 'Compound', 'Sentiment']] = df['review'].apply(analyze_sentiment)

            st.success("Sentiment analysis complete!")
            st.subheader("🔍 Analyzed Reviews")
            st.dataframe(df[['review', 'Sentiment', 'Negative', 'Neutral', 'Positive', 'Compound']])

            # Download results
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)
            st.download_button("📥 Download Result CSV", csv, "review_sentiment_output.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Section 2: Manual User Input ---
st.header("✍️ Analyze Your Own Review")

user_input = st.text_area("Enter your review here:")

if user_input:
    user_scores = sia.polarity_scores(user_input)
    compound = user_scores['compound']
    if compound >= 0.05:
        user_sentiment = "Positive"
    elif compound <= -0.05:
        user_sentiment = "Negative"
    else:
        user_sentiment = "Neutral"

    st.markdown("### 🧠 Sentiment Analysis Result")
    st.write(f"*Sentiment:* {user_sentiment}")
    st.write(f"*Scores:*")
    st.json(user_scores)