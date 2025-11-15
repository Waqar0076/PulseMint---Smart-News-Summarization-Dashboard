# PulseMint---Smart-News-Summarization-Dashboard
PulseMint is a smart AI-style news dashboard built with Streamlit. It supports extractive summarization, advanced filtering, favorites, analytics, dark mode, and user authentication — all using lightweight Python &amp; custom NLP. A full-featured Streamlit application for news browsing, summarization, analytics, and personalization.

This version of PulseMint runs without external AI models, using custom-built extractive NLP and data-scoring methods — making it lightweight, fast, and deployable anywhere.

**Features**
_🔐 User Authentication System_****
Login & Registration system
Admin approval workflow
JSON-based user database

**_⭐ Personalized Experience_**

Add/remove article favorites
View your own favorites dashboard
Dark mode support

_🤖 AI-Style Extractive Summaries_****

Two summarization modes:
Extractive Smart Summaries
Sentence scoring
Frequency-based NLP
Position-awareness
Bullet Point Summary
Automatically converts key sentences into bullets

__**📊 Interactive Analytics (Plotly)**___

Articles by language
Articles by category
Top news sources
Favorites analytics

**📄 Rich Article Viewer**__

Full article content
Clean summarization blocks
Source links
Content length metadata
Pagination & sorting system

**🔍 Advanced Filters**__

Search titles and content in real-time
Filter by language, category, and site
Filter by content length
Show only favorites

**📁 Support for External JSON Uploads**__

Upload your own scraped dataset (formatted like **enhanced_news_scraping_results.json**).

**🏗️ Tech Stack**

** Component     | Technology        **                           

**Frontend      | Streamlit                                   
Backend       | Python                                       
Data Storage  | Local JSON files                             
NLP           | Custom frequency-based extractive summarizer 
Visualization | Plotly Express                               
Others        | Pandas, Regex, Collections**                   

**Project Structure**
InsightBot/
│
├── enhanced_news_scraping_results.json   # Main scraped dataset
├── users.json                            # Authentication database
├── user_favorites.json                   # Favorites tracking
├── app.py                                # Main Streamlit application
└── README.md

**⚙️ Installation & Setup**
**Clone the repository**
git clone https://github.com/yourusername/insightbot.git
cd insightbot

**Install dependencies**
pip install -r requirements.txt

**Run the Streamlit app**
streamlit run app.py

**Requirements**
streamlit==1.40.0
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
python-dateutil==2.9.0.post0
regex==2024.9.11
requests==2.32.3
watchdog==4.0.2
altair==5.3.0
