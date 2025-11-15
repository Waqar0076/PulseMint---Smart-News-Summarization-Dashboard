# import os
# os.environ["TRANSFORMERS_BACKEND"] = "pt"

# import streamlit as st
# from transformers import pipeline
# import json
# import pandas as pd
# from datetime import datetime
# import re
# from io import StringIO
# import subprocess
# import sys
# import torch

# import json, os


# USER_DATA_FILE = "users.json"

# def load_user_data():
#     if not os.path.exists(USER_DATA_FILE):
#         with open(USER_DATA_FILE, "w") as f:
#             json.dump({"users": {}, "pending_users": {}}, f)
#     with open(USER_DATA_FILE, "r") as f:
#         return json.load(f)

# def save_user_data(data):
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(data, f, indent=4)


# # --- 1. Core AI Summarization Logic & Helpers ---
# # Add a try-except block to gracefully handle installation and import issues
# try:
#     from transformers import pipeline
# except ImportError as e:
#     st.error(f"""
#         **ERROR: The 'transformers' library is not installed correctly.**
        
#         This is required for the AI summarization feature. Please fix your environment by running the following command in your terminal:
        
#         `pip install transformers torch sentencepiece`
        
#         After running the command, please restart the Streamlit app.
        
#         *Detailed Error: {e}*
#     """)
#     st.stop()

# # --- Page Configuration and Custom CSS ---
# st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")

# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #2E86AB;
#         font-size: 2.8rem;
#         margin-bottom: 1rem;
#         font-weight: 600;
#     }
#     .subtitle {
#         text-align: center;
#         color: #666;
#         font-size: 1.2rem;
#         margin-bottom: 3rem;
#         font-style: italic;
#     }
#     .summary-box {
#         background: #f0f2f6;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #2E86AB;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background: white;
#         padding: 15px;
#         border-radius: 8px;
#         text-align: center;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- Mock Database & Session State ---
# def initialize_session_state():
#     """Initializes session state variables if they don't exist."""
#     if 'users' not in st.session_state:
#         st.session_state.users = {
#             "admin": {"password": "admin", "role": "admin"},
#             "user": {"password": "user", "role": "user"}
#         }
#     if 'pending_users' not in st.session_state:
#         st.session_state.pending_users = {}
#     if 'logged_in' not in st.session_state:
#         st.session_state.logged_in = False
#     if 'username' not in st.session_state:
#         st.session_state.username = ""
#     if 'role' not in st.session_state:
#         st.session_state.role = ""
#     if 'user_processed_data' not in st.session_state:
#         st.session_state.user_processed_data = None


# # --- AI Model Loading and Helper Functions ---


# @st.cache_resource
# def load_summarizer():
#     """Loads the summarization model (torch-free)."""
#     model_name = "facebook/bart-large-cnn"  # You can also use "sshleifer/distilbart-cnn-12-6" for faster performance
#     try:
#         with st.spinner(f"Loading AI model '{model_name}'..."):
#             summarizer = pipeline("summarization", model=model_name, framework="pt", device=-1)
#         return summarizer
#     except Exception:
#         st.warning("Torch backend not available, switching to TensorFlow/CPU.")
#         summarizer = pipeline("summarization", model=model_name, framework="tf", device=-1)
#         return summarizer



# def summarize_with_ai(text, summarizer):
#     """Generates a summary using the T5 model."""
#     if not text or len(text.split()) < 30:
#         return "Article content is too short to summarize."
#     try:
#         text_to_summarize = "summarize: " + text
#         summary_list = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
#         return summary_list[0]['summary_text']
#     except Exception as e:
#         return f"Could not generate an AI summary. Error: {e}"

# def categorize_article(summary, title=""):
#     """Categorizes articles using keyword matching."""
#     text_to_analyze = (summary + " " + title).lower()
#     categories = {
#         "Technology": ["tech", "ai", "software", "apple", "google"], "Business": ["business", "economy", "market", "stock"],
#         "Politics": ["politics", "government", "election", "trump"], "World News": ["world", "war", "conflict", "ukraine", "gaza"],
#         "Health": ["health", "medical", "disease", "vaccine", "cdc"], "Sports": ["sports", "football", "fifa"],
#         "Science": ["science", "research", "space", "nasa"], "Entertainment": ["entertainment", "movie", "music", "celebrity"]
#     }
#     scores = {cat: sum(1 for kw in kws if kw in text_to_analyze) for cat, kws in categories.items()}
#     if any(s > 0 for s in scores.values()):
#         return max(scores, key=scores.get)
#     return "General"

# def process_uploaded_file(uploaded_file, summarizer):
#     """Processes a user-uploaded JSON file."""
#     try:
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         data = json.load(stringio)
#     except Exception as e:
#         st.error(f"Error reading or parsing the JSON file: {e}")
#         return None

#     all_articles = [article for articles_list in data.get('scraped_data', {}).values() for article in articles_list]
    
#     if not all_articles:
#         st.error("No articles found in the 'scraped_data' section of the file.")
#         return None

#     progress_bar = st.progress(0, text="Processing articles...")
#     for i, article in enumerate(all_articles):
#         content = article.get('content', '')
#         title = article.get('title', '')
#         article['summary'] = summarize_with_ai(content, summarizer)
#         article['category'] = categorize_article(article['summary'], title)
#         source_site = next((k for k, v in data.get('scraped_data', {}).items() if article in v), "Unknown")
#         article['source_site'] = source_site
#         progress_bar.progress((i + 1) / len(all_articles), text=f"Processing: {title[:40]}...")
    
#     progress_bar.empty()
#     st.success(f"✅ Successfully processed {len(all_articles)} articles!")
#     return all_articles


# # --- UI Rendering Functions ---

# def render_user_dashboard():
#     """Renders the main news dashboard for regular users."""
#     st.markdown("<h1 class='main-header'>🤖 InsightBot: Daily News Simplified</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='subtitle'>Your intelligent dashboard to summarize and analyze news.</p>", unsafe_allow_html=True)

#     with st.expander("📁 Summarize Your Own Data"):
#         uploaded_file = st.file_uploader("Upload your JSON dataset", type="json")
#         if uploaded_file:
#             summarizer = load_summarizer()
#             st.session_state.user_processed_data = process_uploaded_file(uploaded_file, summarizer)

#     articles_to_display = st.session_state.user_processed_data
#     if not articles_to_display:
#         try:
#             with open('enhanced_news_scraping_results.json', 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 articles_to_display = [article for articles_list in data.get('scraped_data', {}).values() for article in articles_list]
#                 for article in articles_to_display:
#                     article['source_site'] = next((k for k, v in data.get('scraped_data', {}).items() if article in v), "Unknown")
#         except FileNotFoundError:
#             st.info("Default data not found. Please upload a file or ask an admin to run a scrape.")
#             return
#         except json.JSONDecodeError:
#             st.warning("Default data file is corrupted. Please upload a file or ask an admin to run a scrape.")
#             return

#     if articles_to_display:
#         df = pd.DataFrame(articles_to_display)
#         # Ensure 'summary' and 'category' exist
#         if 'summary' not in df.columns:
#             st.warning("Data needs to be processed. Please upload a file and click 'Process'.")
#             return
        
#         st.header("🔍 Filter & View Articles")
#         # Filters and display logic from standalone app
#         col1, col2 = st.columns(2)
#         with col1:
#             languages = ['All'] + sorted(df['language'].unique().tolist())
#             lang_choice = st.selectbox("Filter by Language:", languages)
#         with col2:
#             categories = ['All'] + sorted(df['category'].unique().tolist())
#             cat_choice = st.selectbox("Filter by Category:", categories)

#         filtered_df = df.copy()
#         if lang_choice != 'All': filtered_df = filtered_df[filtered_df['language'] == lang_choice]
#         if cat_choice != 'All': filtered_df = filtered_df[filtered_df['category'] == cat_choice]

#         if not filtered_df.empty:
#             for _, row in filtered_df.iterrows():
#                 st.markdown(f"### {row.get('title', 'No Title')}")
#                 st.markdown(f"*Source:* `{row.get('source_site', 'N/A')}` | *Language:* `{row.get('language', 'N/A')}`")
#                 with st.expander(f"**{row.get('category', 'General')}** - Read AI-Generated Summary"):
#                     st.markdown(f"<div class='summary-box'>{row.get('summary', 'No summary available.')}</div>", unsafe_allow_html=True)
#                 if row.get('url'): st.markdown(f"[📖 Read Full Article]({row.get('url')})")
#                 st.markdown("---")
#         else:
#             st.info("No articles match your filters.")

# def render_admin_panel():
#     """Renders the admin control panel."""
#     st.markdown("<h1 class='main-header'>⚙️ InsightBot Admin Panel</h1>", unsafe_allow_html=True)

#     st.subheader("📊 Scraper Status")
#     try:
#         with open('enhanced_news_scraping_results.json', 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             summary = data.get('scraping_summary', {})
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Successful Scrapes", summary.get('successful_sites', 'N/A'))
#             col2.metric("Failed Scrapes", summary.get('failed_sites', 'N/A'))
#             col3.metric("Total Articles", summary.get('total_articles_scraped', 'N/A'))
#             with st.expander("View Failed Sites"):
#                 st.json(summary.get('failed_site_list', []))
#     except FileNotFoundError:
#         st.warning("Scraper has not been run yet. Click below to start.")
#     except Exception as e:
#         st.error(f"Could not read scrape data: {e}")

#     st.subheader("🕹️ Scraper Control")
#     if st.button("🚀 Run Web Scraper", help="This will run the index.py script."):
#         with st.spinner("Scraping in progress... This may take several minutes."):
#             try:
#                 process = subprocess.run([f"{sys.executable}", "index.py"], capture_output=True, text=True, check=True)
#                 st.success("Scraper finished successfully!")
#                 st.code(process.stdout)
#             except subprocess.CalledProcessError as e:
#                 st.error("Scraper failed to run.")
#                 st.code(e.stderr)
#             except FileNotFoundError:
#                 st.error("`index.py` not found in the same directory.")

#     st.subheader("🗂️ Data Management")
#     try:
#         with open('enhanced_news_scraping_results.json', "rb") as fp:
#             st.download_button(label="📥 Download JSON Data", data=fp, file_name="scraped_data.json", mime="application/json")
#     except FileNotFoundError:
#         st.info("No data file to download.")

#     st.subheader("👥 User Management (Admin Approval)")
#     if st.session_state.pending_users:
#         st.write("The following users are awaiting approval:")
#         for username, info in st.session_state.pending_users.items():
#             col1, col2 = st.columns([3, 1])
#             col1.text(f"Username: {username}")
#             if col2.button("Approve", key=f"approve_{username}"):
#                 st.session_state.users[username] = {"password": info["password"], "role": "user"}
#                 del st.session_state.pending_users[username]
#                 st.success(f"User '{username}' approved!")
#                 st.rerun()
#     else:
#         st.info("No pending user registrations.")

# def render_login_page():
#     """Renders the login and registration forms."""
#     st.markdown("<h1 class='main-header'>Welcome to InsightBot</h1>", unsafe_allow_html=True)
    
#     choice = st.radio("Select Action", ["Login", "Register"], horizontal=True)

#     if choice == "Login":
#         with st.form("login_form"):
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             submitted = st.form_submit_button("Login")
#             if submitted:
#                 user_data = load_user_data()

#     # Check if the user exists and the password matches
#                 if username in user_data["users"] and user_data["users"][username]["password"] == password:
#                     st.session_state.logged_in = True
#                     st.session_state.username = username
#                     st.session_state.role = user_data["users"][username]["role"]
#                     st.success("✅ Logged in successfully!")
#                     st.rerun()

#     # Handle pending approvals
#                 elif username in user_data["pending_users"]:
#                     st.warning("⏳ Your account is still pending admin approval. Please wait for approval.")

#     # Invalid credentials
#                 else:
#                     st.error("❌ Invalid username or password.")

    
#     elif choice == "Register":
#         with st.form("register_form"):
#             username = st.text_input("Choose a Username")
#             password = st.text_input("Choose a Password", type="password")
#             submitted = st.form_submit_button("Register")
#             if submitted:
#                 user_data = load_user_data()
#                 if username in user_data["users"] or username in user_data["pending_users"]:
#                     st.error("Username already exists or is pending approval.")
#             else:
#                 user_data["pending_users"][username] = {"password": password}
#                 save_user_data(user_data)
#                 st.success("Registration successful! Your account is pending admin approval.")



# # --- Main Application Logic ---
# def main():
#     initialize_session_state()

#     if not st.session_state.logged_in:
#         render_login_page()
#     else:
#         st.sidebar.title(f"Welcome, {st.session_state.username}!")
#         st.sidebar.markdown(f"**Role:** `{st.session_state.role}`")
#         if st.sidebar.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.username = ""
#             st.session_state.role = ""
#             st.session_state.user_processed_data = None
#             st.rerun()
        
#         if st.session_state.role == 'admin':
#             render_admin_panel()
#         else: # 'user' role
#             render_user_dashboard()

# if __name__ == "__main__":
#     main()


# ------------------?

# import streamlit as st
# import json
# import pandas as pd
# import os
# import re

# # ---------------------------------
# # CONFIGURATION
# # ---------------------------------
# st.set_page_config(page_title="InsightBot", page_icon="📰", layout="wide")

# USER_DATA_FILE = "users.json"
# DATA_FILE = "enhanced_news_scraping_results.json"

# # ---------------------------------
# # HELPER FUNCTIONS
# # ---------------------------------
# def load_user_data():
#     if not os.path.exists(USER_DATA_FILE):
#         with open(USER_DATA_FILE, "w") as f:
#             json.dump({"users": {}, "pending_users": {}}, f)
#     with open(USER_DATA_FILE, "r") as f:
#         return json.load(f)

# def save_user_data(data):
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(data, f, indent=4)

# def smart_text_summarize(text, max_length=300):
#     """
#     Smart text summarization without AI - uses simple text processing
#     """
#     if not text:
#         return "No content available"
    
#     # Clean the text
#     text = re.sub(r'\s+', ' ', text.strip())
    
#     # If text is already short, return as is
#     if len(text) <= max_length:
#         return text
    
#     # Try to find a good breaking point (end of sentence)
#     truncated = text[:max_length]
    
#     # Look for the last sentence end within the truncated text
#     last_period = truncated.rfind('. ')
#     last_question = truncated.rfind('? ')
#     last_exclamation = truncated.rfind('! ')
    
#     # Find the latest sentence end
#     sentence_ends = [pos for pos in [last_period, last_question, last_exclamation] if pos != -1]
    
#     if sentence_ends:
#         # Use the latest sentence end
#         cutoff = max(sentence_ends) + 1
#         return truncated[:cutoff] + ".."
#     else:
#         # No sentence end found, just truncate at word boundary
#         last_space = truncated.rfind(' ')
#         if last_space != -1:
#             return truncated[:last_space] + "..."
#         else:
#             return truncated + "..."

# # ---------------------------------
# # AUTHENTICATION
# # ---------------------------------
# def render_login_page():
#     st.title("🔐 InsightBot Login")
#     choice = st.radio("Select Action", ["Login", "Register"], horizontal=True)

#     if choice == "Login":
#         with st.form("login_form"):
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             submit = st.form_submit_button("Login")
#         if submit:
#             data = load_user_data()
#             if username in data["users"] and data["users"][username]["password"] == password:
#                 st.session_state["logged_in"] = True
#                 st.session_state["username"] = username
#                 st.session_state["role"] = data["users"][username]["role"]
#                 st.success("✅ Logged in successfully!")
#                 st.rerun()
#             elif username in data["pending_users"]:
#                 st.warning("⏳ Account pending admin approval.")
#             else:
#                 st.error("❌ Invalid username or password.")
#     else:
#         with st.form("register_form"):
#             username = st.text_input("Choose Username")
#             password = st.text_input("Choose Password", type="password")
#             submit = st.form_submit_button("Register")
#         if submit:
#             if not username or not password:
#                 st.warning("Please fill in all fields.")
#             else:
#                 data = load_user_data()
#                 if username in data["users"] or username in data["pending_users"]:
#                     st.error("Username already exists.")
#                 else:
#                     data["pending_users"][username] = {"password": password}
#                     save_user_data(data)
#                     st.success("✅ Registration successful! Awaiting admin approval.")

# # ---------------------------------
# # USER DASHBOARD (MAIN CONTENT)
# # ---------------------------------
# def render_user_dashboard():
#     st.title("📰 InsightBot — Simplified News Dashboard")
#     st.caption(f"Welcome back, {st.session_state['username']}!")

#     # Sidebar
#     st.sidebar.header("Filters & Settings")
#     if st.sidebar.button("🚪 Logout"):
#         st.session_state["logged_in"] = False
#         st.session_state["username"] = ""
#         st.session_state["role"] = ""
#         st.rerun()

#     # Load Data
#     if not os.path.exists(DATA_FILE):
#         st.warning("⚠️ No data file found. Please contact the admin or upload manually.")
#         uploaded = st.file_uploader("📂 Upload News JSON file", type="json")
#         if uploaded:
#             with open(DATA_FILE, "wb") as f:
#                 f.write(uploaded.read())
#             st.success("✅ File uploaded successfully. Please refresh.")
#         return

#     try:
#         with open(DATA_FILE, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         scraped_data = data.get("scraped_data", {})
#     except Exception as e:
#         st.error(f"Error loading data file: {e}")
#         return

#     rows = []
#     for site, articles in scraped_data.items():
#         for article in articles:
#             rows.append({
#                 "title": article.get("title", "Untitled"),
#                 "content": article.get("content", ""),
#                 "language": article.get("language", "Unknown"),
#                 "category": article.get("category", "Uncategorized"),
#                 "url": article.get("url", ""),
#                 "site": site
#             })

#     df = pd.DataFrame(rows)

#     # Filters
#     st.sidebar.markdown("---")
#     search_query = st.sidebar.text_input("🔍 Search Article")
#     language = st.sidebar.selectbox("🌐 Language", ["All"] + sorted(df["language"].unique()))
#     category = st.sidebar.selectbox("📂 Category", ["All"] + sorted(df["category"].unique()))

#     filtered = df.copy()
#     if search_query:
#         filtered = filtered[filtered["title"].str.contains(search_query, case=False, na=False)]
#     if language != "All":
#         filtered = filtered[filtered["language"] == language]
#     if category != "All":
#         filtered = filtered[filtered["category"] == category]

#     st.markdown("### 📄 Article Summaries")

#     if filtered.empty:
#         st.warning("No matching articles found.")
#     else:
#         for i, row in filtered.iterrows():
#             with st.container():
#                 st.markdown(f"#### 🗞️ {row['title']}")
                
#                 # Generate smart summary without AI
#                 summary = smart_text_summarize(row["content"])
#                 st.write(summary)
                
#                 # Show content length info
#                 content_length = len(row["content"])
#                 if content_length > 1000:
#                     st.caption(f"📏 Full article: {content_length} characters")
                
#                 st.markdown(
#                     f"**Language:** `{row['language']}` | **Category:** `{row['category']}` | "
#                     f"[🔗 Source]({row['url']})"
#                 )
#                 st.markdown("---")

#     # Visualization Section
#     st.markdown("## 📊 Article Insights")
#     if not df.empty:
#         col1, col2 = st.columns(2)
#         with col1:
#             fig1 = px.pie(df, names="language", title="Articles by Language", hole=0.4)
#             st.plotly_chart(fig1, use_container_width=True)
#         with col2:
#             fig2 = px.bar(df["category"].value_counts().reset_index(),
#                           x="index", y="category", title="Articles by Category",
#                           labels={"index": "Category", "category": "Count"})
#             st.plotly_chart(fig2, use_container_width=True)
        
#         # Additional stats
#         col3, col4 = st.columns(2)
#         with col3:
#             st.metric("Total Articles", len(df))
#         with col4:
#             st.metric("Unique Sources", df["site"].nunique())

# # ---------------------------------
# # MAIN FLOW
# # ---------------------------------
# def main():
#     if "logged_in" not in st.session_state:
#         st.session_state["logged_in"] = False
#     if not st.session_state["logged_in"]:
#         render_login_page()
#     else:
#         render_user_dashboard()

# if __name__ == "__main__":
#     main()




# ------------------?


import streamlit as st
import json
import pandas as pd
import os
import re
import plotly.express as px
from datetime import datetime
from collections import Counter
import math

# ---------------------------------
# CONFIGURATION
# ---------------------------------
st.set_page_config(page_title="InsightBot", page_icon="📰", layout="wide")

USER_DATA_FILE = "users.json"
DATA_FILE = "enhanced_news_scraping_results.json"
FAVORITES_FILE = "user_favorites.json"

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------
def load_user_data():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            json.dump({"users": {}, "pending_users": {}}, f)
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_favorites():
    if not os.path.exists(FAVORITES_FILE):
        return {}
    with open(FAVORITES_FILE, "r") as f:
        return json.load(f)

def save_favorites(favorites):
    with open(FAVORITES_FILE, "w") as f:
        json.dump(favorites, f, indent=4)

def toggle_favorite(username, article_id):
    favorites = load_favorites()
    if username not in favorites:
        favorites[username] = []
    
    if article_id in favorites[username]:
        favorites[username].remove(article_id)
    else:
        favorites[username].append(article_id)
    
    save_favorites(favorites)
    return article_id in favorites[username]

def is_favorite(username, article_id):
    favorites = load_favorites()
    return username in favorites and article_id in favorites[username]

def generate_article_id(article):
    """Generate unique ID for each article"""
    return f"{article['site']}_{article['title']}_{article['url']}"

def extractive_summarize(text, num_sentences=3):
    """
    Advanced extractive summarization using sentence scoring
    Based on word frequency and position
    """
    if not text or text.strip() == "":
        return "⚠️ No content available for summarization."
    
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Tokenize and calculate word frequencies
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'for', 'on', 'with', 
                  'as', 'was', 'at', 'by', 'an', 'be', 'this', 'which', 'or', 'from', 'are', 
                  'has', 'had', 'have', 'but', 'not', 'they', 'his', 'her', 'their', 'been',
                  'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall'}
    
    words = [w for w in words if w not in stop_words]
    
    if not words:
        return ' '.join(sentences[:num_sentences])
    
    # Calculate word frequencies
    word_freq = Counter(words)
    max_freq = max(word_freq.values()) if word_freq else 1
    
    # Normalize frequencies
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq
    
    # Score sentences
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        sentence_words = [w for w in sentence_words if w not in stop_words]
        
        if sentence_words:
            # Score based on word frequency
            score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
            
            # Boost score for first few sentences (usually more important)
            if i < 3:
                score *= 1.5
            
            # Penalize very short or very long sentences
            sentence_length = len(sentence.split())
            if sentence_length < 5 or sentence_length > 40:
                score *= 0.5
            
            sentence_scores[i] = score
    
    # Get top N sentences by score, but maintain original order
    top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    top_sentence_indices.sort()  # Maintain original order
    
    # Build summary
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    return ' '.join(summary_sentences)

def generate_bullet_summary(text, num_points=3):
    """
    Generate bullet point summary
    """
    if not text or text.strip() == "":
        return ["No content available"]
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= num_points:
        return sentences
    
    # Score sentences (simplified version)
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        score = len(words)  # Longer sentences often more informative
        
        # Boost first sentences
        if i < 2:
            score *= 1.5
        
        sentence_scores.append((score, i, sentence))
    
    # Get top sentences
    sentence_scores.sort(reverse=True)
    top_sentences = sentence_scores[:num_points]
    
    # Sort by original position
    top_sentences.sort(key=lambda x: x[1])
    
    return [s[2] for s in top_sentences]

# ---------------------------------
# AUTHENTICATION
# ---------------------------------
def render_login_page():
    st.title("🔐 InsightBot Login")
    choice = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    if choice == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            data = load_user_data()
            if username in data["users"] and data["users"][username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = data["users"][username]["role"]
                st.success("✅ Logged in successfully!")
                st.rerun()
            elif username in data["pending_users"]:
                st.warning("⏳ Account pending admin approval.")
            else:
                st.error("❌ Invalid username or password.")
    else:
        with st.form("register_form"):
            username = st.text_input("Choose Username")
            password = st.text_input("Choose Password", type="password")
            submit = st.form_submit_button("Register")
        if submit:
            if not username or not password:
                st.warning("Please fill in all fields.")
            else:
                data = load_user_data()
                if username in data["users"] or username in data["pending_users"]:
                    st.error("Username already exists.")
                else:
                    data["pending_users"][username] = {"password": password}
                    save_user_data(data)
                    st.success("✅ Registration successful! Awaiting admin approval.")

# ---------------------------------
# USER DASHBOARD (MAIN CONTENT)
# ---------------------------------
def render_user_dashboard():
    # Apply dark mode if enabled
    if st.session_state.get('dark_mode', False):
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Fix text visibility in dark mode */
        div[data-testid="stMarkdownContainer"] p {
            color: #FAFAFA !important;
        }
        div[data-testid="stMarkdownContainer"] li {
            color: #FAFAFA !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("📰 InsightBot — Smart News Dashboard")
    st.caption(f"Welcome back, {st.session_state['username']}! 👋")

    # Enhanced Sidebar with all features
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Dark Mode Toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get('dark_mode', False))
        st.session_state['dark_mode'] = dark_mode
        
        # Account Section
        st.subheader("🚪 Account")
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["role"] = ""
            st.rerun()
        
        st.markdown("---")
        st.subheader("🤖 Summarization Settings")
        
        # Summary type
        summary_type = st.radio(
            "Summary Type",
            ["Extractive (Smart)", "Bullet Points"],
            help="Choose how articles should be summarized"
        )
        
        # Number of sentences/points
        if summary_type == "Extractive (Smart)":
            num_sentences = st.slider(
                "Number of sentences", 
                min_value=2, 
                max_value=5, 
                value=3,
                help="How many key sentences to extract"
            )
        else:
            num_sentences = st.slider(
                "Number of bullet points", 
                min_value=2, 
                max_value=5, 
                value=3,
                help="How many key points to show"
            )
        
        st.markdown("---")
        st.subheader("🔍 Search & Filter")
        
        # Real-time Search
        search_query = st.text_input(
            "🔎 Search in titles & content",
            placeholder="Enter keywords...",
            help="Search across article titles and content in real-time"
        )
        
        # Multi-column filters
        col1, col2 = st.columns(2)
        with col1:
            languages = ["All"] + sorted(df["language"].unique())
            selected_language = st.selectbox("🌐 Language", languages)
        with col2:
            categories = ["All"] + sorted(df["category"].unique())
            selected_category = st.selectbox("📂 Category", categories)
        
        # Additional filters
        sites = ["All"] + sorted(df["site"].unique())
        selected_site = st.selectbox("📡 News Source", sites)
        
        # Content length filter
        st.markdown("---")
        st.subheader("📏 Content Settings")
        min_length = st.slider(
            "Minimum content length", 
            min_value=0, 
            max_value=5000, 
            value=0, 
            step=100,
            help="Filter out short articles"
        )
        
        # Favorites filter
        show_favorites = st.checkbox("⭐ Show Favorites Only", help="Display only your favorited articles")

    # Apply filters
    filtered = df.copy()
    
    # Real-time search in titles AND content
    if search_query:
        title_mask = filtered["title"].str.contains(search_query, case=False, na=False)
        content_mask = filtered["content"].str.contains(search_query, case=False, na=False)
        filtered = filtered[title_mask | content_mask]
        if len(filtered) > 0:
            st.info(f"🔍 Found {len(filtered)} articles matching '{search_query}'")
    
    # Language filter
    if selected_language != "All":
        filtered = filtered[filtered["language"] == selected_language]
    
    # Category filter
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    
    # Site filter
    if selected_site != "All":
        filtered = filtered[filtered["site"] == selected_site]
    
    # Content length filter
    filtered = filtered[filtered["content_length"] >= min_length]
    
    # Favorites filter
    if show_favorites:
        favorites = load_favorites().get(st.session_state["username"], [])
        filtered = filtered[filtered["article_id"].isin(favorites)]
        if len(filtered) > 0:
            st.success(f"⭐ Showing {len(filtered)} favorite articles")

    # Display Results Header
    st.header("📄 Article Summaries")
    
    # Results counter and sort controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**📊 Showing {len(filtered)} of {len(df)} total articles**")
    with col2:
        sort_by = st.selectbox(
            "Sort by", 
            ["Newest First", "Oldest First", "Title A-Z", "Title Z-A", "Longest Content", "Shortest Content"]
        )
    with col3:
        items_per_page = st.selectbox("Articles per page", [10, 25, 50], index=0)

    # Apply sorting
    if sort_by == "Title A-Z":
        filtered = filtered.sort_values("title")
    elif sort_by == "Title Z-A":
        filtered = filtered.sort_values("title", ascending=False)
    elif sort_by == "Longest Content":
        filtered = filtered.sort_values("content_length", ascending=False)
    elif sort_by == "Shortest Content":
        filtered = filtered.sort_values("content_length", ascending=True)
    else:  # Default sorting (by index)
        filtered = filtered.sort_index(ascending=(sort_by == "Newest First"))

    # Pagination
    total_pages = max(1, (len(filtered) + items_per_page - 1) // items_per_page)
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered))
        current_articles = filtered.iloc[start_idx:end_idx]
        st.caption(f"📄 Page {page} of {total_pages} (articles {start_idx + 1}-{end_idx})")
    else:
        current_articles = filtered

    # Display articles with AI summarization
    if current_articles.empty:
        st.warning("🚫 No articles found matching your criteria. Try adjusting your filters.")
    else:
        for i, row in current_articles.iterrows():
            with st.container():
                # Article header with favorite button
                col_header1, col_header2 = st.columns([4, 1])
                
                with col_header1:
                    st.markdown(f"### {row['title']}")
                with col_header2:
                    # Favorite button
                    article_id = row['article_id']
                    is_fav = is_favorite(st.session_state["username"], article_id)
                    fav_label = "⭐" if is_fav else "☆"
                    if st.button(fav_label, key=f"fav_{article_id}"):
                        toggle_favorite(st.session_state["username"], article_id)
                        st.rerun()
                
                # Article metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.markdown(f"**🌐 Language:** `{row['language']}`")
                # with col_meta2:
                #     st.markdown(f"**📂 Category:** `{row['category']}`")
                with col_meta3:
                    st.markdown(f"**📡 Source:** `{row['site']}`")
                
                # Generate and display AI summary
                st.markdown("---")
                st.subheader("🤖 AI-Generated Summary")
                
                # Get the content
                article_content = str(row["content"]) if pd.notna(row["content"]) else ""
                
                if article_content and article_content.strip():
                    # Generate summary based on user selection
                    if summary_type == "Extractive (Smart)":
                        summary = extractive_summarize(article_content, num_sentences=num_sentences)
                        
                        # Display summary in styled container
                        st.markdown(f"""
                        <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; 
                                    border-left: 5px solid #1f77b4; margin: 10px 0;">
                            <p style="margin: 0; font-size: 15px; line-height: 1.8; color: #2c3e50;">
                                {summary}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Bullet points summary
                        bullet_points = generate_bullet_summary(article_content, num_points=num_sentences)
                        
                        bullet_html = "<ul style='margin: 0; padding-left: 25px;'>"
                        for point in bullet_points:
                            bullet_html += f"<li style='margin: 10px 0; line-height: 1.6;'>{point}</li>"
                        bullet_html += "</ul>"
                        
                        st.markdown(f"""
                        <div style="background-color: #fff4e6; padding: 20px; border-radius: 10px; 
                                    border-left: 5px solid #ff9800; margin: 10px 0;">
                            {bullet_html}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.caption(f"📊 Full article: {row['content_length']} characters | Summarization: {summary_type}")
                else:
                    st.warning("⚠️ No content available for summarization.")
                
                # Show full content expander
                with st.expander("📖 Read Full Article Content"):
                    if article_content and article_content.strip():
                        st.write(article_content)
                    else:
                        st.info("No full content available.")
                
                # Action buttons
                col_actions1, col_actions2 = st.columns([3, 1])
                with col_actions1:
                    if row['url'] and row['url'] != "N/A" and row['url'].strip():
                        st.markdown(f"[🔗 Read Original Article]({row['url']})")
                with col_actions2:
                    if is_favorite(st.session_state["username"], article_id):
                        st.caption("⭐ Favorited")
                
                st.markdown("---")

    # Enhanced Visualization Section
    st.header("📊 News Insights Dashboard")
    
    if not df.empty:
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "🌐 Languages", "📂 Categories", "📡 Sources", "⭐ Favorites"
        ])
        
        with tab1:
            st.subheader("Dashboard Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", len(df))
            with col2:
                st.metric("Unique Sources", df["site"].nunique())
            with col3:
                avg_length = df["content_length"].mean()
                st.metric("Avg Content Length", f"{avg_length:.0f} chars")
            with col4:
                user_favs = len(load_favorites().get(st.session_state["username"], []))
                st.metric("Your Favorites", user_favs)
        
        with tab2:
            col1, col2 = st.columns([2, 1])
            with col1:
                lang_counts = df['language'].value_counts().reset_index()
                lang_counts.columns = ['language', 'count']
                fig_lang = px.pie(lang_counts, values='count', names='language', 
                                 title="📊 Articles by Language", hole=0.4)
                st.plotly_chart(fig_lang, use_container_width=True)
            with col2:
                st.dataframe(lang_counts, use_container_width=True)
        
        with tab3:
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig_cat = px.bar(cat_counts, x='category', y='count',
                           title="📂 Articles by Category",
                           labels={'category': 'Category', 'count': 'Article Count'})
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with tab4:
            source_counts = df['site'].value_counts().reset_index().head(10)
            source_counts.columns = ['source', 'count']
            fig_source = px.bar(source_counts, x='source', y='count',
                              title="📡 Top News Sources",
                              labels={'source': 'News Source', 'count': 'Article Count'})
            st.plotly_chart(fig_source, use_container_width=True)
        
        with tab5:
            favorites = load_favorites().get(st.session_state["username"], [])
            if favorites:
                fav_articles = df[df["article_id"].isin(favorites)]
                st.subheader(f"⭐ Your Favorite Articles ({len(fav_articles)})")
                
                if not fav_articles.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fav_lang_counts = fav_articles['language'].value_counts().reset_index()
                        fav_lang_counts.columns = ['language', 'count']
                        fig_fav_lang = px.pie(fav_lang_counts, values='count', names='language', 
                                             title="Favorites by Language")
                        st.plotly_chart(fig_fav_lang, use_container_width=True)
                    with col2:
                        fav_cat_counts = fav_articles['category'].value_counts().reset_index()
                        fav_cat_counts.columns = ['category', 'count']
                        fig_fav_cat = px.pie(fav_cat_counts, values='count', names='category',
                                            title="Favorites by Category")
                        st.plotly_chart(fig_fav_cat, use_container_width=True)
                    
                    # Show favorite articles list
                    st.subheader("Your Favorite Articles")
                    for i, row in fav_articles.iterrows():
                        with st.container():
                            st.write(f"**{row['title']}**")
                            st.caption(f"Category: {row['category']} | Language: {row['language']} | Source: {row['site']}")
                            st.markdown("---")
            else:
                st.info("💫 You haven't favorited any articles yet. Click the star icon ☆ to add favorites!")

# ---------------------------------
# MAIN FLOW
# ---------------------------------
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    
    # Load data
    global df
    if not os.path.exists(DATA_FILE):
        st.error("❌ Data file not found. Please ensure the news data file exists.")
        return
    
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        scraped_data = data.get("scraped_data", {})
        
        # Prepare DataFrame with article IDs
        rows = []
        for site, articles in scraped_data.items():
            for article in articles:
                article_data = {
                    "title": article.get("title", "Untitled"),
                    "content": article.get("content", ""),
                    "language": article.get("language", "Unknown"),
                    "category": article.get("category", "Uncategorized"),
                    "url": article.get("url", ""),
                    "site": site,
                    "content_length": len(article.get("content", ""))
                }
                article_data["article_id"] = generate_article_id(article_data)
                rows.append(article_data)
        
        df = pd.DataFrame(rows)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return

    if not st.session_state["logged_in"]:
        render_login_page()
    else:
        render_user_dashboard()

if __name__ == "__main__":
    main()
