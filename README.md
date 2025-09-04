# 📈 **Reddit Sentiment Analyzer (Streamlit + HuggingFace AI)**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg?style=flat-square)](https://github.com/AAkinDev/Sentiment_Analyzer)
[![Deploy](https://img.shields.io/badge/deploy-Streamlit%20Cloud-red?style=flat-square&logo=streamlit)](https://streamlit.io/cloud)

[![NLP](https://img.shields.io/badge/NLP-VADER%20Sentiment-blue?style=flat-square&logo=natural-language-processing)](https://github.com/cjhutto/vaderSentiment)
[![Data](https://img.shields.io/badge/Data-Pandas%20%2B%20Matplotlib-blue?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![AI](https://img.shields.io/badge/AI-DistilGPT%2D2-purple?style=flat-square)](https://huggingface.co/distilgpt2)

### 🚀 A fully automated Reddit sentiment analysis dashboard using Streamlit, VADER NLP, and HuggingFace Inference API for intelligent AI summaries and insights.

## 🛠️ **Project Overview**
This project collects real-time Reddit posts from selected subreddits, analyzes their sentiment using NLP (VADER), and visualizes trends in an interactive Streamlit dashboard. AI-generated insights and recommendations are powered by HuggingFace.

## 💡 **Use Cases**
✅ Track sentiment trends across topics like technology, AI, crypto, or politics.  

✅ Monitor brand reputation based on subreddit discussions.

✅ Automate collection, analysis, and visualization of Reddit data for decision-making.

## 📈 Architecture & Tech Stack

| Component              | Technology Used                                         |
|------------------------|---------------------------------------------------------|
| Data Collection        | 🔗 Reddit API via PRAW                                  |
| Sentiment Analysis     | 🧠 VADER SentimentIntensityAnalyzer                     |
| Visualization          | 📊 Streamlit + Matplotlib + Seaborn                     |
| AI Summaries           | 🤖 HuggingFace Inference API (DistilGPT-2)              |
| Env Management         | 🌎 Python-dotenv                                        |
| Deployment (Optional)  | 🌐 Streamlit Cloud / HuggingFace Spaces                 |


## 🛠️ **Features**
✅ Reddit API Integration with PRAW

✅ Real-time Sentiment Analysis using NLP (VADER)

✅ AI-generated Summaries and Actionable Insights

✅ Clean and interactive Dashboard (Streamlit)

✅ CSV Export for further analysis

✅ Beautiful data visualizations (Bar Chart, Pie Chart, WordCloud, Box Plot, Time Series)

## 🔄 **Workflow Breakdown**

### 1️⃣ **Reddit Data Collection**

import praw

reddit = praw.Reddit(client_id=..., client_secret=..., user_agent=...)

subreddit = reddit.subreddit("technology")

for post in subreddit.hot(limit=50):
    title = post.title
    score = post.score


### 2️⃣ **Sentiment Analysis with VADER**

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sentiment_score = analyzer.polarity_scores(title)['compound']

### 3️⃣ AI-Powered Summaries using HuggingFace API

import requests

api_token = "your_huggingface_token"

headers = {"Authorization": f"Bearer {api_token}"}
payload = {"inputs": "Summarize Reddit sentiment analysis results."}

response = requests.post("https://api-inference.huggingface.co/models/distilgpt2", headers=headers, json=payload)
ai_text = response.json()[0]['generated_text']

### 4️⃣ **Visualization with Streamlit**

import streamlit as st

import matplotlib.pyplot as plt

import seaborn as sns

st.title("📈 Reddit Sentiment Analyzer")

st.dataframe(df)

sns.countplot(x='sentiment_label', data=df)

st.pyplot()

## 👨‍💻 How to Run This Project

### 1️⃣ Clone the Repository

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Set Up Environment Variables

Create a .env file in the root directory:

CLIENT_ID=your_reddit_client_id

CLIENT_SECRET=your_reddit_client_secret

USER_AGENT=your_app_user_agent

USERNAME=your_reddit_username

PASSWORD=your_reddit_password

HUGGINGFACE_API_TOKEN=your_huggingface_api_token

### 4️⃣ Run Streamlit Dashboard

streamlit run app.py

📈 **Results & Insights**

## Sample Insights:

- 🚀 AI-related subreddits showed highly positive sentiment.

- 💬 Crypto discussions were more polarized.

- 🔥 Negative sentiment spikes aligned with controversial news events.


## 💪 Future Improvements

✅ Expand to multiple subreddit groups for cross-community analysis

✅ Train a custom ML model for even smarter sentiment detection

✅ Deploy fully to Streamlit Cloud / Hugging Face Spaces

✅ Integrate topic extraction (Named Entity Recognition)

## 📚 Contributions

👥 Contributions are welcome!

### Steps to contribute:

- Fork the repo

- Create a feature branch 

- Submit a Merge Request (MR) with improvements

## 💎 License

📜 MIT License — free to use and modify with attribution.

📱 ### Connect With Me

👨🏽‍💻 :Git:LH @AkSquare_dev @AAkinDev 

🌐 Portfolio: WIP

👤 LinkedIn: AkSq 

## 🔥 Project Badge

Built by Akin A (AkSquare_Dev) 🚀

## 📌 Note

This project is for educational, portfolio, and light production use. For massive-scale deployments, consider adding advanced caching, queueing, and auto-scaling infrastructures.


