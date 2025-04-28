import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud
import requests
from dotenv import load_dotenv
import os
import re
import config
import time

# Helper: clean AI output
def clean_ai_text(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+[a-zA-Z]*\.\.\.$', '', text)
    text = re.sub(r'\s+', ' ', text)
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if not text.endswith('.'):
        text += '.'
    return text.strip()

# Helper: HuggingFace API call
def generate_text_from_huggingface(prompt):
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        raise Exception("HuggingFace API token not found. Please set it in your .env file.")
    
    API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.4
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    generated = response.json()
    return generated[0]['generated_text']

# Streamlit Config
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")

# Load environment variables
load_dotenv()

# Reddit API Setup
Reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
    password=os.getenv("PASSWORD")
)

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# App Title
st.title('📈 Reddit Sentiment Analyzer')

subreddits_input = st.text_input('Enter subreddits (comma-separated if multiple)', config.DEFAULT_SUBREDDITS)
post_limit = st.slider('Number of posts per subreddit', 5, 50, config.DEFAULT_POST_LIMIT)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

if st.button('Analyze') or st.session_state.analyzed:
    st.session_state.analyzed = True

    with st.spinner('Fetching Reddit posts and analyzing sentiment...'):
        subreddits = [sub.strip() for sub in subreddits_input.split(',')]
        posts = []
        for subreddit_name in subreddits:
            subreddit = Reddit.subreddit(subreddit_name)
            for submission in subreddit.hot(limit=post_limit):
                posts.append({
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "sentiment": analyzer.polarity_scores(submission.title)['compound'],
                    "created_utc": pd.to_datetime(submission.created_utc, unit='s')
                })

    time.sleep(0.5)
    st.success('✅ Analysis Complete!')

    df = pd.DataFrame(posts)

    if not df.empty:
        df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))
        sentiment_counts = Counter(df['sentiment_label'])
        sentiment_colors = config.SENTIMENT_COLORS

        selected_sentiment = st.radio("Select Sentiment to View:", ("All", "Positive", "Neutral", "Negative"), horizontal=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Dashboard", "�� AI Insights", "🧠 Actionable Insights"])

        with tab1:
            st.subheader('📊 Sentiment Analysis Overview')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Positive", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
            with col3:
                st.metric("❌ Negative", sentiment_counts.get('Negative', 0))

            if selected_sentiment != "All":
                filtered_df = df[df['sentiment_label'] == selected_sentiment]
            else:
                filtered_df = df

            st.dataframe(filtered_df[['subreddit', 'title', 'sentiment_label', 'created_utc']].sort_values(by='created_utc', ascending=True), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full Data as CSV", data=csv, file_name='reddit_sentiment_analysis.csv', mime='text/csv')

        with tab2:
            st.subheader('📈 📊 Reddit Sentiment Analysis Dashboard')
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_bar, ax_bar = plt.subplots(figsize=(4,3))
                sns.countplot(x='sentiment_label', data=df, order=['Positive', 'Neutral', 'Negative'], palette=[sentiment_colors[s] for s in ['Positive', 'Neutral', 'Negative']], ax=ax_bar)
                ax_bar.set_title('📊 Distribution of Sentiments')
                ax_bar.set_xlabel('Sentiment')
                ax_bar.set_ylabel('Number of Posts')
                st.pyplot(fig_bar)
            with col2:
                fig_pie, ax_pie = plt.subplots(figsize=(4,3))
                ax_pie.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), colors=[sentiment_colors[s] for s in sentiment_counts.keys()], autopct='%1.1f%%', startangle=140)
                ax_pie.axis('equal')
                ax_pie.set_title('Sentiment Share')
                st.pyplot(fig_pie)
            with col3:
                text = " ".join(post['title'] for post in posts)
                wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Set2').generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(6,4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title('🔥 Top Words Cloud')
                st.pyplot(fig_wc)

            st.subheader('Sentiment Spread - Box Plot and 📊 Sentiment Breakdown')
            col_box, col_hbar = st.columns(2)
            with col_box:
                fig_box, ax_box = plt.subplots(figsize=(5,4))
                sns.boxplot(x='sentiment_label', y='sentiment', data=df, hue='sentiment_label', dodge=False, palette=sentiment_colors, order=['Positive', 'Neutral', 'Negative'], ax=ax_box, legend=False)
                ax_box.set_ylabel('Sentiment Score')
                ax_box.set_xlabel('Sentiment')
                ax_box.set_title('Box Plot of Sentiment Scores')
                st.pyplot(fig_box)
            with col_hbar:
                fig_hbar, ax_hbar = plt.subplots(figsize=(5,4))
                sns.barplot(y=list(sentiment_counts.keys()), x=list(sentiment_counts.values()), palette=[sentiment_colors[s] for s in sentiment_counts.keys()], ax=ax_hbar)
                ax_hbar.set_title('Horizontal Sentiment Counts')
                ax_hbar.set_xlabel('Number of Posts')
                ax_hbar.set_ylabel('Sentiment')
                st.pyplot(fig_hbar)

            st.subheader('⏱️ Sentiment Over Time - 📈 Time Series')
            df['date'] = df['created_utc'].dt.date
            timeseries = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
            fig_ts, ax_ts = plt.subplots(figsize=(8,5))
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in timeseries.columns:
                    ax_ts.plot(timeseries.index, timeseries[sentiment], label=sentiment, color=sentiment_colors[sentiment])
            ax_ts.set_title('Sentiment Trends Over Time')
            ax_ts.set_xlabel('Date')
            ax_ts.set_ylabel('Number of Posts')
            ax_ts.legend()
            st.pyplot(fig_ts)

        with tab3:
            st.subheader('🧠 AI-Generated Insights')
            if st.button('Generate AI Summary', key='ai_summary'):
                try:
                    prompt = (
                        f"You are a market analyst. Based on the Reddit sentiment analysis: "
                        f"{sentiment_counts.get('Positive', 0)} positive, "
                        f"{sentiment_counts.get('Neutral', 0)} neutral, and "
                        f"{sentiment_counts.get('Negative', 0)} negative posts. "
                        "Generate a concise insight or recommendation."
                    )
                    ai_text = generate_text_from_huggingface(prompt)
                    st.write(clean_ai_text(ai_text))
                except Exception as e:
                    st.error(f"⚠️ AI Summary generation failed: {str(e)}")

        with tab4:
            st.subheader('🧠 Generate Actionable Insights')
            user_prompt = st.text_input('Ask for an insight based on the analysis:', 'Suggest an action based on the current sentiment trends.')
            if st.button('Generate Insight', key='actionable_insight'):
                try:
                    full_prompt = (
                        f"Sentiment counts: Positive={sentiment_counts.get('Positive',0)}, "
                        f"Neutral={sentiment_counts.get('Neutral',0)}, Negative={sentiment_counts.get('Negative',0)}. "
                        f"{user_prompt}"
                    )
                    ai_text = generate_text_from_huggingface(full_prompt)
                    st.write(clean_ai_text(ai_text))
                except Exception as e:
                    st.error(f"⚠️ Insight generation failed: {str(e)}")
else:
    st.warning('⚠️ No data yet. Please click Analyze to start.')

# Footer
st.markdown("""
---
<div style='text-align: center; font-size: 1.0em;'>
    By Akin A (AkSquare_Dev)
</div>
""", unsafe_allow_html=True)

