import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Setup
st.set_page_config(page_title="Navigent BI Dashboard", layout="wide")
st.title("Navigent BI Dashboard (Exam Project)")
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
filtered_df = df[df['target_industry'].isin(selected_industry)]

# 4. KPI Kort
st.subheader("Overordnede KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Totale Kampagner", len(filtered_df))
col2.metric("Møder Booket (Total)", int(filtered_df['meetings_booked'].sum()))
col3.metric("Gns. AI Fit Score", f"{filtered_df['avg_ai_fit_score'].mean():.1f}")

# 5. Interaktiv Graf
st.subheader("Performance: Standard vs. Deep Dive")
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=filtered_df, x='enrichment_mode', y='booking_rate_pct', palette='viridis', ax=ax, errorbar=None)
ax.set_ylabel("Booking Rate (%)")
ax.set_xlabel("Berigelsesmetode")
st.pyplot(fig)

st.markdown("---")
st.markdown("*Udviklet til Business Intelligence eksamen. Data er syntetisk genereret baseret på Navigents struktur.*")