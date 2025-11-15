import streamlit as st
import json
import os
import subprocess
import sys
import pandas as pd
import plotly.express as px

# ---------------------------------
# CONFIG
# ---------------------------------
st.set_page_config(page_title="InsightBot Admin Panel", page_icon="🧠", layout="wide")
USER_DATA_FILE = "users.json"
DATA_FILE = "enhanced_news_scraping_results.json"

# ---------------------------------
# Helper Functions
# ---------------------------------
def load_user_data():
    """Load user and pending user data."""
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            json.dump({"users": {}, "pending_users": {}}, f)
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_user_data(data):
    """Save user data back to the file."""
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def login_admin():
    """Admin login screen."""
    st.title("🔐 Admin Login")
    st.write("Please log in with your administrator credentials to access the control panel.")
    with st.form("admin_login"):
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        user_data = load_user_data()
        if username in user_data["users"]:
            user = user_data["users"][username]
            if user["password"] == password and user["role"] == "admin":
                st.session_state["admin_logged_in"] = True
                st.session_state["admin_username"] = username
                st.success("✅ Welcome back, Admin!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials or insufficient privileges.")
        else:
            st.error("❌ Admin account not found.")

# ---------------------------------
# Dashboard Overview
# ---------------------------------
def render_dashboard():
    st.header("📊 Admin Dashboard Overview")
    user_data = load_user_data()

    # --- USER METRICS ---
    pending_users = user_data.get("pending_users", {})
    approved_users = user_data.get("users", {})
    total_pending = len(pending_users)
    total_users = len(approved_users)
    total_admins = sum(1 for u in approved_users.values() if u.get("role") == "admin")

    # --- ARTICLE METRICS ---
    total_articles = 0
    articles_df = pd.DataFrame()
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                scraped_data = data.get("scraped_data", {})
                all_articles = []
                for site, articles in scraped_data.items():
                    for a in articles:
                        all_articles.append({
                            "site": site,
                            "language": a.get("language", "Unknown"),
                            "category": a.get("category", "Uncategorized"),
                            "date": a.get("date", None)
                        })
                articles_df = pd.DataFrame(all_articles)
                total_articles = len(articles_df)
        except Exception as e:
            st.error(f"Error reading articles data: {e}")

    # --- METRIC CARDS ---
    st.markdown("### 🧮 Key System Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Users", total_users)
    c2.metric("🕒 Pending Requests", total_pending)
    c3.metric("🧑‍💼 Admin Accounts", total_admins)
    c4.metric("📰 Total Articles", total_articles)

    st.markdown("---")

    # --- PIE CHART: USER DISTRIBUTION ---
    st.markdown("### 👥 User Overview")
    user_chart_data = pd.DataFrame({
        "Type": ["Approved Users", "Pending Users"],
        "Count": [total_users, total_pending]
    })
    fig1 = px.pie(user_chart_data, names="Type", values="Count", color="Type",
                  color_discrete_map={"Approved Users": "#2E86DE", "Pending Users": "#F39C12"},
                  title="User Distribution Overview", hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)

    # --- ARTICLE CHARTS ---
    if not articles_df.empty:
        st.markdown("### 🗞️ Article Analytics")

        col1, col2 = st.columns(2)
        with col1:
            lang_counts = articles_df["language"].value_counts().reset_index()
            lang_counts.columns = ["Language", "Count"]
            fig2 = px.bar(lang_counts, x="Language", y="Count", color="Language",
                          title="Articles by Language", text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            cat_counts = articles_df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig3 = px.bar(cat_counts, x="Category", y="Count", color="Category",
                          title="Articles by Category", text_auto=True)
            st.plotly_chart(fig3, use_container_width=True)

        # Time-series if 'date' is available
        if "date" in articles_df.columns and articles_df["date"].notna().any():
            st.markdown("### ⏱️ Daily Article Volume")
            try:
                articles_df["date"] = pd.to_datetime(articles_df["date"], errors="coerce")
                daily = articles_df.groupby(articles_df["date"].dt.date).size().reset_index(name="Articles")
                fig4 = px.line(daily, x="date", y="Articles", markers=True,
                               title="Articles Published Over Time")
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.warning("Could not parse dates for timeline chart.")

    else:
        st.info("No article data found yet. Run the scraper to populate insights.")

# ---------------------------------
# User Management
# ---------------------------------
def render_user_management():
    st.header("👥 User Management")
    user_data = load_user_data()
    pending_users = user_data.get("pending_users", {})
    approved_users = user_data.get("users", {})

    st.subheader("🕒 Pending User Requests")

    if pending_users:
        for username, info in pending_users.items():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"**👤 {username}** — awaiting approval")

                if col2.button("✅ Approve", key=f"approve_{username}"):
                    user_data["users"][username] = {"password": info["password"], "role": "user"}
                    del user_data["pending_users"][username]
                    save_user_data(user_data)
                    st.success(f"✅ User '{username}' approved successfully!")
                    st.rerun()

                if col3.button("❌ Deny", key=f"deny_{username}"):
                    del user_data["pending_users"][username]
                    save_user_data(user_data)
                    st.warning(f"🚫 User '{username}' request denied.")
                    st.rerun()
    else:
        st.info("No pending user requests.")

    st.markdown("---")
    st.subheader("✅ Approved Users")
    if approved_users:
        df = pd.DataFrame([
            {"Username": u, "Role": info.get("role", "user")}
            for u, info in approved_users.items()
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No approved users available.")


# ---------------------------------
# Scraper Control
# ---------------------------------
def render_scraper_control():
    st.header("🕹️ Scraper Control Panel")
    st.info("Run your `index.py` web scraper script directly from here.")
    if st.button("🚀 Run Web Scraper"):
        with st.spinner("Running scraper... Please wait..."):
            try:
                process = subprocess.run([sys.executable, "index.py"], capture_output=True, text=True, check=True)
                st.success("✅ Web scraping completed successfully!")
                st.code(process.stdout)
            except subprocess.CalledProcessError as e:
                st.error("❌ Web scraping failed.")
                st.code(e.stderr)
            except FileNotFoundError:
                st.error("⚠️ `index.py` not found in this directory.")


# ---------------------------------
# Data Management
# ---------------------------------
def render_data_management():
    st.header("🗂️ Data Management & Export")
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            st.download_button("📥 Download Scraped Data", f, file_name="scraped_data.json", mime="application/json")
    else:
        st.info("⚠️ No scraped data available for download.")
    st.markdown("---")
    st.caption("Use this section to download or manage collected news data.")


# ---------------------------------
# Admin Panel Layout
# ---------------------------------
def render_admin_dashboard():
    st.sidebar.title(f"👋 Welcome, {st.session_state.get('admin_username', 'Admin')}")
    st.sidebar.markdown("---")
    option = st.sidebar.radio(
        "Admin Sections",
        ["🏠 Dashboard", "👥 User Management", "📊 Scraper Control", "🗂️ Data Management"]
    )
    if st.sidebar.button("🚪 Logout"):
        st.session_state["admin_logged_in"] = False
        st.session_state["admin_username"] = ""
        st.rerun()

    if option == "🏠 Dashboard":
        render_dashboard()
    elif option == "👥 User Management":
        render_user_management()
    elif option == "📊 Scraper Control":
        render_scraper_control()
    elif option == "🗂️ Data Management":
        render_data_management()

# ---------------------------------
# MAIN
# ---------------------------------
def main():
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    if not st.session_state["admin_logged_in"]:
        login_admin()
    else:
        render_admin_dashboard()


if __name__ == "__main__":
    main()
