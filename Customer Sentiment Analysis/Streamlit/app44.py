# app4_final.py - dynamic Plotly version (run: streamlit run app4_final.py)
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Sentiment Analysis📊",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DARK MODE THEME
# ============================================================
st.markdown("""
<style>
.stApp { background:#0e1117; color:#f1f5f9; }
[data-testid="stSidebar"] { background:#111827; border-right:1px solid #263244; }
[data-testid="stHeader"] { background:#0e1117; }
.stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6 { color:#f1f5f9 !important; }
[data-testid="stMetric"] { background:#151b26; border:1px solid #2b3545; border-radius:12px; padding:12px; }
[data-testid="stDataFrame"] { border:1px solid #2b3545; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# Static dark Matplotlib/Seaborn charts
plt.rcParams.update({
    "figure.facecolor":"#0e1117", "axes.facecolor":"#151b26",
    "axes.edgecolor":"#475569", "axes.labelcolor":"#f1f5f9",
    "axes.titlecolor":"#f1f5f9", "xtick.color":"#cbd5e1",
    "ytick.color":"#cbd5e1", "text.color":"#f1f5f9",
    "legend.facecolor":"#151b26", "legend.edgecolor":"#475569",
    "grid.color":"#334155", "grid.alpha":0.35
})


# ============================================================
# LOGIN / REGISTRATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_DB_PATH = PROJECT_ROOT / "Data" / "users.csv"
LOGIN_LOG_PATH = PROJECT_ROOT / "Data" / "login_activity.csv"


def ensure_user_db():
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USER_DB_PATH.exists():
        pd.DataFrame([
            {"username": "admin", "password": "admin123"}
        ]).to_csv(USER_DB_PATH, index=False)


def load_users():
    ensure_user_db()
    return pd.read_csv(USER_DB_PATH)


def save_users(user_df):
    user_df.to_csv(USER_DB_PATH, index=False)


def ensure_login_log():
    LOGIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOGIN_LOG_PATH.exists():
        pd.DataFrame(columns=["username", "timestamp", "status"]).to_csv(LOGIN_LOG_PATH, index=False)


def log_login_activity(username, status="success"):
    ensure_login_log()
    log_df = pd.read_csv(LOGIN_LOG_PATH)
    new_entry = pd.DataFrame([
        {
            "username": str(username).strip(),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status
        }
    ])
    log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_df.to_csv(LOGIN_LOG_PATH, index=False)


def load_login_activity(limit=10):
    ensure_login_log()
    log_df = pd.read_csv(LOGIN_LOG_PATH)
    return log_df.tail(limit).reset_index(drop=True)


def display_login_activity_sidebar():
    st.subheader("📝 Recent Login Activity")
    activity = load_login_activity(5)
    if activity.empty:
        st.caption("No login activity yet.")
    else:
        for _, row in activity.iterrows():
            status = row["status"]
            label = (
                "✅ Success" if status == "success"
                else "📝 Registered" if status == "registered"
                else "❌ Failed" if status == "failed"
                else "🚪 Logout"
            )
            st.caption(f"{label} • {row['username']} • {row['timestamp']}")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🔐 Customer Sentiment Analysis Login")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            users = load_users()
            matching_users = users[users["username"].astype(str) == username.strip()]
            if matching_users.empty:
                log_login_activity(username, "failed")
                st.error("Username not found.")
            elif str(matching_users.iloc[0]["password"]) != password:
                log_login_activity(username, "failed")
                st.error("Incorrect password.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                log_login_activity(username, "success")
                st.rerun()

        st.info("Default login: username `admin` | password `admin123`")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("New username")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            registered = st.form_submit_button("Create account")

        if registered:
            username_value = new_username.strip()
            users = load_users()
            if not username_value or not new_password:
                st.error("Enter a username and password.")
            elif users["username"].astype(str).eq(username_value).any():
                st.error("That username already exists.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = pd.concat(
                    [
                        users,
                        pd.DataFrame(
                            [{"username": username_value, "password": new_password}]
                        ),
                    ],
                    ignore_index=True,
                )
                save_users(users)
                log_login_activity(username_value, "registered")
                st.success("Account created. You can now log in.")

    st.stop()

with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.get('username', 'User')}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        log_login_activity(st.session_state.get('username', 'User'), "logout")
        st.session_state.username = ""
        st.rerun()


# ============================================================
# DATA LOADING
# ============================================================
def resolve_dataset_path(base_dir):
    data_dir = base_dir / "Data"
    candidates = [
        data_dir / "Ecommerce_Customer_Sentiment_csv.csv",
        data_dir / "Ecommerce_Customer_Sentiment_Cleaned.csv",
        data_dir / "Ecommerce_Customer_Sentiment.csv",
        data_dir / "Ecommerce_Customer_Sentiment.csv.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(data_dir.glob("*Ecommerce_Customer_Sentiment*.csv*"))
    if matches:
        return matches[0]

    project_matches = sorted(base_dir.glob("**/*Ecommerce_Customer_Sentiment*.csv*"))
    if project_matches:
        return project_matches[0]

    raise FileNotFoundError(
        f"Default dataset not found in {data_dir}. Expected one of: "
        + ", ".join(str(path.name) for path in candidates)
    )


DEFAULT_PATH = resolve_dataset_path(PROJECT_ROOT)


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        default_file = DEFAULT_PATH if DEFAULT_PATH.exists() else resolve_dataset_path(Path.cwd())
        if not default_file.exists():
            raise FileNotFoundError(f"Default dataset not found: {default_file}")
        data = pd.read_csv(default_file)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    data = data.drop_duplicates()

    if "review_date" in data.columns:
        data["review_date"] = pd.to_datetime(
            data["review_date"],
            errors="coerce",
            dayfirst=True
        )

    if "rating" in data.columns:
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")

    if "sentiment" in data.columns:
        data["sentiment"] = (
            data["sentiment"]
            .astype(str)
            .str.strip()
            .str.title()
        )

    if "sentiment_score" in data.columns:
        data["sentiment_score"] = pd.to_numeric(
            data["sentiment_score"], errors="coerce"
        )

    if "purchase_amount" in data.columns:
        data["purchase_amount"] = pd.to_numeric(
            data["purchase_amount"], errors="coerce"
        )

    if "complaint_type" in data.columns:
        data["complaint_type"] = (
            data["complaint_type"]
            .astype(str)
            .str.strip()
            .replace({"nan": "No Complaint", "None": "No Complaint"})
        )

    return data

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Settings")

# Dataset is loaded automatically from the project's Data folder.
try:
    df = load_data()
except Exception as e:
    st.error("❌ Dataset could not be loaded.")
    st.info(f"Check that your CSV exists at:\n{DEFAULT_PATH}")
    st.exception(e)
    st.stop()

st.sidebar.success(f"Dataset loaded: {len(df):,} reviews")

# ============================================================
# PAGE NAVIGATION
# ============================================================
menu = st.sidebar.radio(
    "📌 Select Analysis",
    [
        "🏠 Dashboard",
        "😊 Sentiment Analysis",
        "⭐ Rating Analysis",
        "📦 Product Analysis",
        "😡 Complaint Analysis",
        "🏷️ Brand Analysis",
        "👥 Customer Analysis",
        "📍 State & City Analysis",
        "💳 Payment & Channel",
        "🤖 Rating Prediction",
        "📄 Data Table"
    ]
)

# ============================================================
# 🎛️ DYNAMIC FILTERS (slicers) - shown ONLY on the pages listed here.
# To show them on more pages, add the page name to FILTER_PAGES.
# ============================================================
FILTER_PAGES = ["🏠 Dashboard"]

filtered_df = df.copy()

if menu in FILTER_PAGES:
    st.sidebar.markdown("### 🎛️ Dynamic Filters")

    # --- State slicer ---
    if "state" in df.columns:
        states = sorted(df["state"].dropna().astype(str).unique().tolist())
        selected_state = st.sidebar.selectbox("🌎 State", ["All"] + states, key="slicer_state")
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df["state"].astype(str) == selected_state]

    # --- City slicer (options follow the selected state) ---
    if "city" in df.columns:
        cities = sorted(filtered_df["city"].dropna().astype(str).unique().tolist())
        # reset a stale city choice when the state changes
        if st.session_state.get("slicer_city", "All") not in ["All"] + cities:
            st.session_state["slicer_city"] = "All"
        selected_city = st.sidebar.selectbox("🏙️ City", ["All"] + cities, key="slicer_city")
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df["city"].astype(str) == selected_city]

    # --- Purchase Channel slicer ---
    if "purchase_channel" in df.columns:
        preferred = ["Marketplace", "Website", "Mobile App"]
        present = df["purchase_channel"].dropna().astype(str).str.strip().unique().tolist()
        channel_options = (
            ["All"]
            + [c for c in preferred if c in present]
            + sorted(c for c in present if c not in preferred)
        )
        selected_channel = st.sidebar.selectbox("🛒 Purchase Channel", channel_options, key="slicer_channel")
        if selected_channel != "All":
            filtered_df = filtered_df[
                filtered_df["purchase_channel"].astype(str).str.strip() == selected_channel
            ]

    # --- Sentiment slicer ---
    if "sentiment" in df.columns:
        selected_sentiment = st.sidebar.selectbox(
            "😊 Sentiment", ["All", "Positive", "Negative", "Neutral"], key="slicer_sentiment"
        )
        if selected_sentiment != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    st.sidebar.caption(f"Showing {len(filtered_df):,} of {len(df):,} reviews")

analysis_df = filtered_df

# ============================================================
# HELPER
# ============================================================
def available(col):
    return col in analysis_df.columns

def safe_mean(col):
    if available(col):
        return analysis_df[col].mean()
    return 0


# ============================================================
# REUSABLE CHART FUNCTIONS (interactive Plotly, controlled by widgets)
# ============================================================
PLOTLY_CONFIG = {
    "scrollZoom": True,          # mouse wheel = zoom in / zoom out
    "displayModeBar": True,      # always show toolbar (zoom +, zoom -, pan, reset)
    "displaylogo": False,
    "doubleClick": "reset",      # double-click = reset the view
    "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d"],
}

SENT_COLORS = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}


def show(fig, height=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#151b26",
                      height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      dragmode="zoom", hovermode="closest")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def dist_chart(col, title, key):
    """Count of reviews per value of `col`, with Bar/Pie toggle."""
    if not available(col):
        st.info(f"{col} column is not available.")
        return
    kind = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, key=f"{key}_kind")
    counts = analysis_df[col].value_counts().rename_axis(col).reset_index(name="Reviews")
    if counts.empty:
        st.info("No data for the current filters.")
        return
    if col == "rating":
        counts = counts.sort_values(col)
        counts[col] = counts[col].astype(str)
    cmap = SENT_COLORS if col == "sentiment" else None
    if kind == "Pie":
        fig = px.pie(counts, names=col, values="Reviews", hole=0.35,
                     color=col, color_discrete_map=cmap, title=title)
    else:
        fig = px.bar(counts, x=col, y="Reviews", color=col, color_discrete_map=cmap,
                     text_auto=True, title=title)
        fig.update_layout(showlegend=False)
    show(fig)


def ranked_bar(series, title, label, key):
    """Top/bottom-N horizontal bar chart from a Series (index -> value)."""
    series = series.dropna()
    if series.empty:
        st.info("No data for the current filters.")
        return
    c1, c2 = st.columns(2)
    n = c1.slider("Show items", 1, len(series), min(10, len(series)), key=f"{key}_n") if len(series) > 1 else 1
    order = c2.radio("Order", ["Highest", "Lowest"], horizontal=True, key=f"{key}_o")
    series = series.nlargest(n) if order == "Highest" else series.nsmallest(n)
    d = series.reset_index()
    d.columns = ["Name", label]
    fig = px.bar(d, x=label, y="Name", orientation="h", text_auto=".2f", title=title)
    fig.update_yaxes(categoryorder="total ascending", title="")
    show(fig)


def avg_ranked(col, val, title, key):
    """Average of `val` per `col`, with a minimum-reviews filter."""
    if not (available(col) and available(val)):
        st.info(f"{col} or {val} column is not available.")
        return
    g = analysis_df.groupby(col)[val].agg(["mean", "count"])
    if g.empty:
        st.info("No data for the current filters.")
        return
    min_n = st.slider("Minimum reviews per item", 1, max(int(g["count"].max()), 2), 1, key=f"{key}_min")
    ranked_bar(g[g["count"] >= min_n]["mean"], title, f"Average {val}", key)


def crosstab_chart(col, title, key):
    """Sentiment breakdown per `col` with Top-N and Grouped/Stacked/100% controls."""
    if not (available(col) and available("sentiment")):
        st.warning(f"{col} or sentiment column is not available.")
        return
    ct = pd.crosstab(analysis_df[col], analysis_df["sentiment"])
    if ct.empty:
        st.info("No data for the current filters.")
        return
    c1, c2 = st.columns(2)
    n = c1.slider("Show top N", 1, len(ct), min(10, len(ct)), key=f"{key}_n") if len(ct) > 1 else 1
    mode = c2.radio("Bar mode", ["Grouped", "Stacked", "100%"], horizontal=True, key=f"{key}_m")
    ct = ct.loc[ct.sum(axis=1).nlargest(n).index]
    long = ct.reset_index().melt(id_vars=col, var_name="Sentiment", value_name="Reviews")
    fig = px.bar(long, x=col, y="Reviews", color="Sentiment", color_discrete_map=SENT_COLORS,
                 barmode="group" if mode == "Grouped" else "relative", title=title)
    if mode == "100%":
        fig.update_layout(barnorm="percent")
        fig.update_yaxes(title="Share of reviews (%)")
    show(fig)
    with st.expander("View table"):
        st.dataframe(ct, use_container_width=True)


def trend_chart():
    """Time trend with selectable period and metric."""
    if not (available("review_date") and available("sentiment")):
        return
    temp = analysis_df.dropna(subset=["review_date"]).copy()
    if temp.empty:
        return
    c1, c2 = st.columns(2)
    label = c1.radio("Group by", ["Day", "Week", "Month"], index=2, horizontal=True, key="trend_f")
    metrics = ["Review count"] + (["Average rating"] if available("rating") else [])
    metric = c2.radio("Metric", metrics, horizontal=True, key="trend_m")
    temp["period"] = temp["review_date"].dt.to_period({"Day": "D", "Week": "W", "Month": "M"}[label]).dt.start_time
    if metric == "Review count":
        d = temp.groupby(["period", "sentiment"]).size().reset_index(name="Reviews")
        fig = px.line(d, x="period", y="Reviews", color="sentiment", color_discrete_map=SENT_COLORS,
                      markers=True, title=f"{label}ly Sentiment Trend")
    else:
        d = temp.groupby("period")["rating"].mean().reset_index()
        fig = px.line(d, x="period", y="rating", markers=True, title=f"{label}ly Average Rating")
    show(fig, 450)



# ============================================================
# DASHBOARD
# ============================================================
if menu == "🏠 Dashboard":

    st.markdown("""
        <h1 style="font-size:42px;font-weight:700;margin-bottom:5px;">
        🛒 E-Commerce Customer Sentiment Analysis
        </h1>
        <p style="color:#9ca3af;font-size:16px;margin-top:0;">
        Customer reviews, sentiment, satisfaction, product and sales analytics
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style="font-size:32px;margin-top:35px;margin-bottom:25px;">
        🏠 Dashboard
        </h2>
    """, unsafe_allow_html=True)

    # KPI CARDS
    total_reviews = len(analysis_df)
    avg_rating = (
        analysis_df["rating"].mean()
        if "rating" in analysis_df.columns and total_reviews > 0
        else 0
    )
    avg_purchase = (
        analysis_df["purchase_amount"].mean()
        if "purchase_amount" in analysis_df.columns and total_reviews > 0
        else 0
    )
    negative_reviews = (
        (analysis_df["sentiment"].astype(str).str.lower() == "negative").sum()
        if "sentiment" in analysis_df.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total_reviews:,}")
    c2.metric("Average Rating", f"{avg_rating:.2f}")
    c3.metric("Average Purchase", f"₹{avg_purchase:,.2f}")
    c4.metric("Negative Reviews", f"{negative_reviews:,}")

    st.divider()

    col1, col2 = st.columns(2)

    # OVERALL CUSTOMER SENTIMENT
    with col1:
        st.subheader("Overall Customer Sentiment")

        if "sentiment" in analysis_df.columns:
            sentiment_counts = (
                analysis_df["sentiment"]
                .value_counts()
                .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
                .reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_sentiment = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count"
            )
            fig_sentiment.update_traces(
                textinfo="percent",
                textposition="inside"
            )
            fig_sentiment.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=430,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_sentiment, use_container_width=True, config=PLOTLY_CONFIG)

    # RATING DISTRIBUTION
    with col2:
        st.subheader("Rating Distribution")

        if "rating" in analysis_df.columns:
            rating_counts = (
                analysis_df["rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            rating_counts.columns = ["rating", "count"]

            fig_rating = px.bar(
                rating_counts,
                x="rating",
                y="count",
                text="count"
            )
            fig_rating.update_traces(textposition="inside")
            fig_rating.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=430,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="rating",
                yaxis_title="count"
            )
            st.plotly_chart(fig_rating, use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================
elif menu == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    if not available("sentiment"):
        st.warning("sentiment column is not available.")
        st.stop()

    counts = analysis_df["sentiment"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", int(counts.get("Positive", 0)))
    c2.metric("Neutral", int(counts.get("Neutral", 0)))
    c3.metric("Negative", int(counts.get("Negative", 0)))

    dist_chart("sentiment", "Customer Sentiment Distribution", "sent_dist")

    if available("rating"):
        st.subheader("Average Rating by Sentiment")
        ranked_bar(analysis_df.groupby("sentiment")["rating"].mean(),
                   "Average Rating by Sentiment", "Average rating", "sent_avg")

    if available("rating") and available("sentiment_score") and len(analysis_df):
        st.subheader("Rating vs Sentiment Score")
        sample = analysis_df.dropna(subset=["rating", "sentiment_score"])
        sample = sample.sample(min(len(sample), 3000), random_state=1)
        fig = px.strip(sample, x="rating", y="sentiment_score", color="sentiment",
                       color_discrete_map=SENT_COLORS, title="Rating vs Sentiment Score (sample)")
        show(fig)

# ============================================================
# RATING ANALYSIS
# ============================================================
elif menu == "⭐ Rating Analysis":
    st.header("⭐ Rating Analysis")
    if not available("rating") or analysis_df.empty:
        st.warning("rating column is not available or no data matches the filters.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rating", f"{analysis_df['rating'].mean():.2f}")
    c2.metric("Highest Rating", f"{analysis_df['rating'].max():.0f}")
    c3.metric("Lowest Rating", f"{analysis_df['rating'].min():.0f}")

    dist_chart("rating", "Rating Distribution", "rate_dist")

    if available("sentiment"):
        st.subheader("Rating by Sentiment")
        table = analysis_df.groupby("sentiment")["rating"].agg(["count", "mean"])
        table.columns = ["Reviews", "Average Rating"]
        st.dataframe(table.round(2), use_container_width=True)

# ============================================================
# PRODUCT ANALYSIS
# ============================================================
elif menu == "📦 Product Analysis":
    st.header("📦 Product Analysis")
    if not available("product_name"):
        st.warning("product_name column is not available.")
        st.stop()

    st.subheader("Products by Average Rating")
    avg_ranked("product_name", "rating", "Products by Average Rating", "prod_rate")

    if available("purchase_amount"):
        st.subheader("Revenue and Purchases")
        rc, pc = st.columns(2)
        with rc:
            ranked_bar(analysis_df.groupby("product_name")["purchase_amount"].sum(),
                       "Total Revenue by Product", "Revenue", "prod_rev")
        with pc:
            ranked_bar(analysis_df.groupby("product_name").size(),
                       "Purchases by Product", "Purchases", "prod_cnt")

    st.subheader("Product-wise Sentiment")
    crosstab_chart("product_name", "Product-wise Sentiment", "prod_sent")

# ============================================================
# COMPLAINT ANALYSIS
# ============================================================
elif menu == "😡 Complaint Analysis":
    st.header("😡 Customer Complaint Analysis")
    if not available("complaint_type"):
        st.warning("complaint_type column is not available.")
        st.stop()

    only_real = st.checkbox("Exclude 'No Complaint'", value=True)
    comp_df = analysis_df
    if only_real:
        comp_df = analysis_df[analysis_df["complaint_type"].astype(str).str.casefold() != "no complaint"]
    complaints = comp_df["complaint_type"].value_counts()
    if complaints.empty:
        st.info("No complaints for the current filters.")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Complaint Types", complaints.shape[0])
    c2.metric("Most Common Complaint", str(complaints.index[0]))

    ranked_bar(complaints, "Customer Complaints", "Complaints", "comp")
    st.subheader("Complaint Type vs Sentiment")
    crosstab_chart("complaint_type", "Complaint Type vs Sentiment", "comp_sent")

# ============================================================
# BRAND ANALYSIS
# ============================================================
elif menu == "🏷️ Brand Analysis":
    st.header("🏷️ Brand-wise Sentiment")
    if not available("brand"):
        st.warning("brand column is not available.")
        st.stop()

    crosstab_chart("brand", "Brand-wise Sentiment", "brand_sent")
    st.subheader("Brands by Average Rating")
    avg_ranked("brand", "rating", "Brands by Average Rating", "brand_rate")

# ============================================================
# CUSTOMER ANALYSIS
# ============================================================
elif menu == "👥 Customer Analysis":
    st.header("👥 Customer Analysis")
    st.subheader("Customer Type vs Sentiment")
    crosstab_chart("customer_type", "Customer Type vs Sentiment", "cust_sent")

    if available("rating"):
        st.subheader("Customer Satisfaction")
        threshold = st.slider("Satisfied if rating is at least", 1, 5, 4)
        sat = np.where(analysis_df["rating"] >= threshold, "Satisfied", "Not Satisfied")
        satisfaction = pd.Series(sat).value_counts()
        c1, c2 = st.columns(2)
        c1.metric("Satisfied", int(satisfaction.get("Satisfied", 0)))
        c2.metric("Not Satisfied", int(satisfaction.get("Not Satisfied", 0)))
        d = satisfaction.rename_axis("Satisfaction").reset_index(name="Customers")
        show(px.pie(d, names="Satisfaction", values="Customers", hole=0.4, title="Customer Satisfaction"))

# ============================================================
# STATE & CITY ANALYSIS (one combined page)
# ============================================================
elif menu == "📍 State & City Analysis":
    st.header("📍 State & City Analysis")
    st.caption("Charts are interactive: scroll to zoom in/out, drag to pan, double-click to reset.")

    if not available("state") or not available("sentiment") or analysis_df.empty:
        st.warning("state/sentiment column is not available or no data matches the filters.")
    else:
        # ---------------- STATE SECTION ----------------
        st.subheader("📍 State Analysis")
        state_counts = analysis_df["state"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("States Covered", analysis_df["state"].nunique())
        c2.metric("Top State", state_counts.index[0])
        c3.metric("Top State Reviews", f"{state_counts.iloc[0]:,}")

        crosstab_chart("state", "State-wise Sentiment", "state_sent")

        if available("rating"):
            st.markdown("**States by Average Rating**")
            avg_ranked("state", "rating", "States by Average Rating", "state_rate")

        st.divider()

        # ---------------- CITY SECTION ----------------
        st.subheader("🏙️ City Analysis")
        full_df = analysis_df
        drill_state = st.selectbox(
            "Drill down to a state",
            ["All"] + sorted(full_df["state"].dropna().astype(str).unique().tolist()),
            key="sc_state"
        )
        if drill_state != "All":
            # helper charts read the global analysis_df, so narrow it for the city section
            analysis_df = full_df[full_df["state"].astype(str) == drill_state]

        if not available("city") or analysis_df.empty:
            st.info("No city data for the current selection.")
        else:
            city_counts = analysis_df["city"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Cities Covered", analysis_df["city"].nunique())
            c2.metric("Top City", city_counts.index[0])
            c3.metric("Top City Reviews", f"{city_counts.iloc[0]:,}")

            st.markdown("**City Review Volume**")
            ranked_bar(city_counts, "Cities by Review Count", "Reviews", "city_vol")
            st.markdown("**City-wise Sentiment**")
            crosstab_chart("city", "City-wise Sentiment", "city_sent")
            st.markdown("**Cities by Average Rating**")
            avg_ranked("city", "rating", "Cities by Average Rating", "city_rate")
            st.markdown("**Sentiment Score by City**")
            avg_ranked("city", "sentiment_score", "Sentiment Score by City", "city_score")

            # statistical summary per city
            if available("rating"):
                with st.expander("📈 Statistical summary by city"):
                    agg = {"rating": ["count", "mean", "median", "std", "min", "max"]}
                    stats = analysis_df.groupby("city").agg(agg)
                    stats.columns = ["Reviews", "Mean", "Median", "Std Dev", "Min", "Max"]
                    st.dataframe(stats.sort_values("Reviews", ascending=False).round(2),
                                 use_container_width=True)

            if available("complaint_type"):
                st.markdown("**Top Complaint Type by City**")
                city_complaints = (
                    analysis_df[analysis_df["complaint_type"].astype(str).str.lower() != "no complaint"]
                    .groupby(["city", "complaint_type"]).size().reset_index(name="count")
                    .sort_values("count", ascending=False).head(10)
                )
                st.dataframe(city_complaints, use_container_width=True)

        analysis_df = full_df

# ============================================================
# PAYMENT & CHANNEL
# ============================================================
elif menu == "💳 Payment & Channel":
    st.header("💳 Payment Method & Purchase Channel")
    col1, col2 = st.columns(2)
    with col1:
        crosstab_chart("payment_method", "Payment Method vs Sentiment", "pay")
    with col2:
        crosstab_chart("purchase_channel", "Purchase Channel vs Sentiment", "chan")

# ============================================================
# ML RATING PREDICTION
# ============================================================
elif menu == "🤖 Rating Prediction":

    st.header("🤖 Rating Prediction using Machine Learning")
    with st.expander("ℹ️ How does Rating Prediction work?", expanded=True):
        st.markdown(
            """
**1. Clean the review text** – each review is lowercased and links, numbers and
special characters are removed.

**2. Convert text to numbers (TF-IDF)** – every review becomes a list of numbers.
Words that are important in a review but rare across all reviews (e.g. *excellent*,
*poor*) get higher weights than common words.

**3. Train / test split** – 80% of the reviews are used to train the model and
20% are kept aside to test it.

**4. Linear Regression** – the model learns which words push the rating up
(e.g. *excellent, love*) and which pull it down (e.g. *poor, bad*).

**5. Predict** – for a new review, the same cleaning and TF-IDF steps are applied,
and the model outputs a rating between 1 and 5. A rating of **4 or more** is
shown as *Satisfied*.

**Model accuracy metrics:** MAE / RMSE = average error in rating points (lower is
better). R² = how much of the rating variation the model explains (closer to 1 is
better).
            """
        )

    required = ["review_text", "rating"]

    if not all(available(c) for c in required):
        st.error("review_text and rating columns are required.")
        st.stop()

    model_df = analysis_df[["review_text", "rating"]].copy()
    model_df["review_text"] = model_df["review_text"].fillna("").astype(str)
    model_df["rating"] = pd.to_numeric(model_df["rating"], errors="coerce")
    model_df = model_df.dropna(subset=["rating"])

    model_df["Processed_Review"] = (
        model_df["review_text"]
        .str.lower()
        .str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
        .str.replace(r"[^a-zA-Z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    X_text = model_df["Processed_Review"]
    y = model_df["rating"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 5)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("R² Score", f"{r2:.2f}")

    st.divider()

    st.subheader("✍️ Predict Rating for a New Review")

    review_input = st.text_area(
        "Enter customer review",
        "The product quality is excellent and delivery was very fast"
    )

    if st.button("🔮 Predict Rating"):
        new_data = vectorizer.transform([review_input])
        prediction = model.predict(new_data)[0]
        prediction = float(np.clip(prediction, 1, 5))

        st.slider(
            "Predicted Rating",
            min_value=1.0,
            max_value=5.0,
            value=prediction,
            step=0.1,
            disabled=True,
            help="Model-predicted rating on a 1 to 5 scale"
        )

        if prediction >= 4:
            st.success("Customer Satisfaction: Satisfied")
        else:
            st.warning("Customer Satisfaction: Not Satisfied")

# ============================================================
# DATA TABLE
# ============================================================
elif menu == "📄 Data Table":

    st.header("📄 Customer Sentiment Data")

    search = st.text_input("🔎 Search in review text")

    display_df = analysis_df.copy()

    if search and available("review_text"):
        display_df = display_df[
            display_df["review_text"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    st.write(f"Showing {len(display_df):,} rows")
    st.dataframe(display_df, use_container_width=True, height=600)

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name="customer_sentiment_cleaned.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Customer Sentiment Analysis | Python • Pandas • Streamlit • Scikit-learn")