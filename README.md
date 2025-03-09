## 📈 Reddit Sentiment Analysis with Pipedream & Streamlit

🚀 Automated Reddit sentiment analysis using Pipedream for data collection, Google Sheets/PostgreSQL for storage, and Streamlit for interactive visualization.

### 🛠️ Project Overview
This project tracks sentiment on Reddit discussions in real-time using Pipedream automation. The data is processed using NLP sentiment analysis, stored in Google Sheets/PostgreSQL, and visualized in a Streamlit dashboard.

### 💡 Use Cases
✅ Track public sentiment on AI, tech trends, crypto, or politics
✅ Monitor brand reputation based on subreddit discussions
✅ Automate data collection for continuous analysis

#### 📈 Architecture & Tech Stack
Component	Technology Used
Data Collection	🛠️ Pipedream (Reddit API)
Sentiment Analysis	🧠 TextBlob, Vader (NLP)

#### Storage	
📊 Google Sheets / PostgreSQL

### Visualization	
📈 Streamlit, Plotly

#### Deployment	
🌐 Streamlit Cloud / Hugging Face Spaces

#### 🛠️ Features
✅ Serverless automation with Pipedream
✅ Reddit API Integration (PRAW, REST API)
✅ Sentiment analysis using NLP (TextBlob, Vader)
✅ Data storage in Google Sheets/PostgreSQL
✅ Interactive Streamlit Dashboard
🔄 Workflow Breakdown

#### 1️⃣ Automating Data Collection with Pipedream

import { axios } from "@pipedream/platform";

export default defineComponent({
  async run({ steps }) {
    const redditUrl = "https://www.reddit.com/r/technology/hot.json?limit=50";
    const response = await axios(this, { method: "GET", url: redditUrl });

    return response.data.data.children.map(post => ({
      title: post.data.title,
      upvotes: post.data.ups
    }));
  }
});

#### 2️⃣ Storing Data in Google Sheets / PostgreSQL
#### Option 1: Google Sheets
  - Pipedream sends the processed Reddit data to Google Sheets using the Google Sheets API.

#### Option 2: PostgreSQL Database

CREATE TABLE reddit_sentiment (
    id SERIAL PRIMARY KEY,
    post_title TEXT,
    sentiment_score FLOAT,
    upvotes INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

### 3️⃣ Visualization with Streamlit

import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Reddit Sentiment Analysis")

### Load data
df = pd.read_csv("reddit_sentiment.csv")

### Sentiment Distribution Plot
fig = px.histogram(df, x="Sentiment", nbins=20, title="Sentiment Analysis")
st.plotly_chart(fig)

### 👨‍💻 How to Run This Project

#### 1️⃣ Clone the Repository
git clone https://gitlab.com/aakdev1/reddit-sentiment-analysis.git
cd reddit-sentiment-analysis

#### 2️⃣ Install Dependencies
pip install praw textblob pandas plotly streamlit

#### 3️⃣ Set Up Environment Variables
Create a `.env` file and add:
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
USER_AGENT=MyRedditSentimentApp

#### 4️⃣ Run Data Collection Script
python fetch_reddit_data.py

#### 5️⃣ Run Streamlit Dashboard
streamlit run app.py


### 📈 Results & Insights
#### Sample Insights from Reddit Analysis:
- AI-related posts received highly positive sentiment.
- Crypto-related discussions were more polarized.
- Negative sentiment spikes corresponded with controversial news.

#### 💪 Future Improvements
✅ Train a custom ML model for more accurate sentiment classification
✅ Expand to multiple subreddits to track cross-community sentiment
✅ Use Named Entity Recognition (NER) to extract key topics
✅ Deploy Streamlit app publicly on Streamlit Cloud / Hugging Face

#### 📚 Contributions
👥 Contributions are welcome! If you’d like to enhance this project:
1. Fork the repo
2. Create a new feature branch
3. Submit a PR with improvements

#### 💎 License
📚 MIT License - Feel free to use and modify this project.

### 📱 Connect With Me
📺 GitHub/GitLab: [@aksquare_dev](https://gitlab.com/aakdev1) / gitlab/aksquare_dev
👤 LinkedIn: to be updated
🌐 Portfolio: to be updated
