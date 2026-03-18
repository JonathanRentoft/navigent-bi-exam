import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Page Setup
st.set_page_config(page_title="Navigent BI Dashboard", layout="wide")
st.title("🚀 Navigent BI Dashboard (Exam Project)")
st.markdown("Interaktiv analyse af kampagne-performance og AI-berigelse.")

# 2. Load and Clean Data
@st.cache_data 
def load_data():
    df = pd.read_csv('NAVIGENT_MOCK_DATA.csv')
    df = df[df['emails_sent'] < 10000] # Fjern outliers
    df['meetings_booked'] = pd.to_numeric(df['meetings_booked'], errors='coerce').fillna(0)
    df['booking_rate_pct'] = (df['meetings_booked'] / df['emails_sent']) * 100
    df['target_industry'] = df['target_industry'].str.lower().str.strip()
    return df

df = load_data()

# 3. Sidebar (Interaktivitet for censor)
st.sidebar.header("Filtrér Data 🔍")
selected_industry = st.sidebar.multiselect(
    "Vælg Branche:",
    options=df['target_industry'].unique(),
    default=df['target_industry'].unique()
)

# Filtrer dataen baseret på brugerens valg
filtered_df = df[df['target_industry'].isin(selected_industry)].copy()

# ---------------------------------------------------------
# OPRET FANER (TABS) FOR AT FÅ DET TIL AT LIGNE EN RIGTIG APP
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Forretnings-KPI'er", "🤖 Machine Learning Insights"])

# --- FANE 1: FORRETNING ---
with tab1:
    st.subheader("Overordnede KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Totale Kampagner (Filtreret)", len(filtered_df))
    col2.metric("Møder Booket (Total)", int(filtered_df['meetings_booked'].sum()))
    col3.metric("Gns. AI Fit Score", f"{filtered_df['avg_ai_fit_score'].mean():.1f}")

    st.markdown("---")
    
    # Vi sætter to grafer ved siden af hinanden i kolonner
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Standard vs. Deep Dive")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='enrichment_mode', y='booking_rate_pct', palette='viridis', ax=ax, errorbar=None)
        ax.set_ylabel("Booking Rate (%)")
        ax.set_xlabel("Berigelsesmetode")
        st.pyplot(fig)
        
    with chart_col2:
        st.subheader("Konvertering pr. Branche")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='target_industry', y='meetings_booked', palette='magma', ax=ax2, errorbar=None)
        ax2.set_ylabel("Gns. Bookede Møder")
        ax2.set_xlabel("Branche")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# --- FANE 2: MACHINE LEARNING ---
with tab2:
    st.subheader("Kundesegmentering (K-Means Clustering)")
    st.markdown("Her kører en Unsupervised Machine Learning model *live* på den filtrerede data for at finde kundesegmenter.")
    
    # Vi kører K-means live på det data, brugeren har filtreret!
    if len(filtered_df) > 10:
        cluster_features = ['emails_sent', 'avg_ai_fit_score']
        X_cluster = filtered_df[cluster_features].dropna()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        filtered_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.scatterplot(data=filtered_df, x='emails_sent', y='avg_ai_fit_score', hue='cluster', palette='Set1', s=80, ax=ax3)
        ax3.set_title('K-Means Clustering: AI Kvalitet vs. Email Volumen')
        st.pyplot(fig3)
    else:
        st.warning("Vælg venligst flere industrier i sidebaren for at lade maskinen bygge klynger.")

    st.markdown("---")
    
    st.subheader("Feature Korrelation (Heatmap)")
    numeric_cols = filtered_df[['emails_sent', 'bounces', 'emails_opened', 'emails_replied', 'meetings_booked', 'avg_ai_fit_score', 'booking_rate_pct']]
    corr = numeric_cols.corr()
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax=ax4)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("*Udviklet til Business Intelligence eksamen. Data er syntetisk genereret baseret på Navigent arkitektur.*")