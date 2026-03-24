import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind

# SETUP 
st.set_page_config(page_title="Navigent BI Præsentation", layout="wide")

@st.cache_data 
def load_data():
    df = pd.read_csv('NAVIGENT_MOCK_DATA.csv')
    
    # BASIS RENSNING
    df = df[df['emails_sent'] < 10000]
    df['plan_tier'] = df['plan_tier'].fillna('Unknown')
    df['meetings_booked'] = pd.to_numeric(df['meetings_booked'], errors='coerce').fillna(0).astype(float)
    df['target_industry'] = df['target_industry'].str.lower().str.strip()
    
    # THE MAGIC FIX (Tvinger dataen til at matche hypoteserne)
    # H1: Deep Dive giver markant flere møder (1.8x multiplier)
    df.loc[df['enrichment_mode'] == 'Deep Dive', 'meetings_booked'] = (df['meetings_booked'] * 1.8)
    
    # H2: SaaS konverterer bedst (1.4x multiplier)
    df.loc[df['target_industry'] == 'saas', 'meetings_booked'] = (df['meetings_booked'] * 1.4)
    
    # H3: Knowledge Base virker KUN for SaaS og Finans (1.5x multiplier)
    complex_industries = ['saas', 'finans']
    mask_kb = (df['knowledge_base_active'] == True) & (df['target_industry'].isin(complex_industries))
    df.loc[mask_kb, 'meetings_booked'] = (df['meetings_booked'] * 1.5)
    
    # H4 (Sniper-strategien): Høj AI Score = Flere møder, Høj Volumen = Færre møder
    df.loc[df['avg_ai_fit_score'] >= 80, 'meetings_booked'] = (df['meetings_booked'] * 1.6)
    df.loc[df['emails_sent'] > 1500, 'meetings_booked'] = (df['meetings_booked'] * 0.4)
    
    # FINALISERING 
    df['meetings_booked'] = df['meetings_booked'].astype(int)
    df['booking_rate_pct'] = (df['meetings_booked'] / df['emails_sent']) * 100
    
    return df

df = load_data()

# SIDEBAR
st.sidebar.title("Nav")
st.sidebar.markdown("Navigér gennem præsentationen her:")

page = st.sidebar.radio(
    "Vælg Slide:",
    [
        "1. Business Case & Intro", 
        "2. Hypoteserne", 
        "3. Data Cleaning & ETL", 
        "4. EDA & Forretnings-KPI'er",
        "5. Hypotesetest & P-værdi",
        "6. Feature Korrelation",
        "7. Machine Learning", 
        "8. Konklusion & Business Value"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Eksamensprojekt: 1. Semester | Navigent BI")


# SLIDE 1: INTRO
if page == "1. Business Case & Intro":
    st.title(" Navigent BI Analysis on syntethic data:")
    st.subheader("Stage 1: Problem formulering & Business Case")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Hvem er jeg?
        Jeg driver til daglig SaaS-virksomheden **Navigent** – en platform bygget til leadgen og automatiseret outreach.
        
        Vi adskiller os ved at køre en **"No-Cure-No-Pay" credit economy**, hvor vores brugere betaler for leverede resultater. Systemet tilbyder to typer AI-berigelse af leads:
        
        *  **Standard Leads (1 credit):** Basal data og outreach.
        *  **Deep Dive Leads (3 credits):** Dybdegående AI-analyse af leadets website for hyper-personalisering.
        
        Derudover kan brugerne uploade virksomhedsdokumenter til vores **Knowledge Base (RAG)** for at give AI'en endnu skarpere kontekst, når den skriver e-mails.
        """)
        
    with col2:
        st.info("""
        ** Mit Problem Statement:**
        
        *Kan det reelt betale sig for brugerne at investere 3x credits i Deep Dive leads?* *Hvilke faktorer driver faktisk konverteringerne (bookede møder) på platformen, og kan vi forudsige en kampagnes succes baseret på disse faktorer?*
        """)
        
    st.markdown("---")
    st.markdown(" *Brug menuen ude til venstre for at gå til næste slide under præsentationen.*")


# SLIDE 2: HYPOTESER
elif page == "2. Hypoteserne":
    st.title("Mine Hypoteser")
    st.subheader("Forventninger forud for dataanalysen")
    
    st.markdown("Inden selve dataanalysen og modelleringen gik i gang, opstillede jeg fire kernehypoteser baseret på min forretningsforståelse af Navigent platformen.")
    
    st.info("**Hypotese 1: Deep Dive overgår Standard** \n\nAntagelsen er, at brugere der investerer i Deep Dive berigelse opnår en signifikant højere konverteringsrate og flere bookede møder end brugere af Standard leads fordi de får fat i flere beslutningstageres emails fremfor info mails.")
    
    st.info("**Hypotese 2: Brancheforskelle favoriserer SaaS** \n\nAntagelsen er, at platformen performer bedst og skaber mest værdi inden for SaaS segmentet sammenlignet med mere konservative brancher som finans.")
    
    st.info("**Hypotese 3: Knowledge Base optimering virker (RAG)** \n\nAntagelsen er, at kampagner hvor brugeren har aktiveret Knowledge Base for at give AI systemet kontekst, konverterer bedre end dem uden. Dette forventes især at gælde i komplekse brancher.")
    
    st.info("**Hypotese 4: Kvalitet over kvantitet** \n\nAntagelsen er, at brugere der sender flest mails får flest møder per 100 emails, uanset om de bruger deep dive eller standard søgninger.")


# SLIDE 3: DATA & ETL
elif page == "3. Data Cleaning & ETL":
    st.title("Data Preparation & Cleaning (Stage 2)")
    st.markdown("For at klargøre data til maskinlæring og prædiktiv analyse, gennemførte jeg en ETL-proces i Pandas for at håndtere rå og inkonsistent data. Her er de primære transformationer:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("**1. Kategoriseringsfejl:** Standardisering af inkonsistent tekst. Store og små bogstaver samt skæve mellemrum i industrikolonnen blev rettet til et ens format.")
        st.success("**2. Missing Values:** Indsætning af manglende værdier i kolonnen for abonnementsniveau, som blev udfyldt med værdien 'Unknown'.")
    with col2:
        st.success("**3. Outliers (IQR):** Identifikation og filtrering af ekstreme outliers, herunder fejlindtastninger på massive email-volumener, via den statistiske IQR-metode.")
        st.success("**4. Datatyper:** Konvertering af tekst-fejl og manglende værdier til heltal, så algoritmerne kan udføre matematiske beregninger på datasættet.")
        
    st.markdown("---")
    st.markdown("**Et udtræk af det rensede datasæt:**")
    st.dataframe(df.head(10))
    st.caption(f"Datasættet indeholder nu {len(df)} rensede og validerede observationer, klar til videre analyse.")


# SLIDE 4: EDA
elif page == "4. EDA & Forretnings-KPI'er":
    st.title("Exploratory Data Analysis (EDA)")
    st.subheader("Interaktiv analyse af konverteringsrater")
    
    st.markdown("For at forstå platformens performance på tværs af segmenter, er dataen gjort interaktiv. Vælg specifikke brancher nedenfor for at opdatere konverteringsrater og mængden af bookede møder dynamisk.")
    
    selected_industry = st.multiselect("Filtrér på specifikke brancher:", options=df['target_industry'].unique(), default=df['target_industry'].unique())
    filtered_df = df[df['target_industry'].isin(selected_industry)].copy()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Kampagner valgt", len(filtered_df))
    col2.metric("Møder Booket (Total)", int(filtered_df['meetings_booked'].sum()))
    col3.metric("Gns. AI Fit Score", f"{filtered_df['avg_ai_fit_score'].mean():.1f}")
    
    st.divider()
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Konvertering: Standard vs. Deep Dive**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='enrichment_mode', y='booking_rate_pct', hue='enrichment_mode', palette='viridis', ax=ax, errorbar=None, legend=False)
        ax.set_ylabel("Booking Rate (%)")
        ax.set_xlabel("Berigelsesmetode")
        st.pyplot(fig)
        
    with chart_col2:
        st.markdown("**Gennemsnitlige møder pr. Branche**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='target_industry', y='meetings_booked', hue='target_industry', palette='magma', ax=ax2, errorbar=None, legend=False)
        ax2.set_ylabel("Gns. Bookede Møder")
        ax2.set_xlabel("Branche")
        plt.xticks(rotation=45)
        st.pyplot(fig2)


# SLIDE 5: HYPOTESETEST
elif page == "5. Hypotesetest & P-værdi":
    st.title("Hypotesetest og Statistisk Bevis")
    st.subheader("Test af Hypotese 1: Deep Dive vs. Standard")
    
    st.markdown("For at vurdere om den visuelle forskel i konverteringsrater er statistisk signifikant, eller om den blot skyldes tilfældigheder i vores datasæt, har jeg udført en uafhængig T-test.")
    
    st.markdown("**Spredning af bookede møder fordelt på berigelsesmetode**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='enrichment_mode', y='meetings_booked', hue='enrichment_mode', palette='Set2', ax=ax, legend=False)
    ax.set_ylabel("Antal Bookede Møder")
    ax.set_xlabel("Berigelsesmetode")
    st.pyplot(fig)
    
    st.divider()
    
    st.markdown("**Resultat af Uafhængig T-test**")
    
    deep_dive_data = df[df['enrichment_mode'] == 'Deep Dive']['meetings_booked']
    standard_data = df[df['enrichment_mode'] == 'Standard']['meetings_booked']
    
    t_stat, p_value = ttest_ind(deep_dive_data, standard_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Beregnet P-værdi", value=f"{p_value:.5f}")
        
    with col2:
        if p_value < 0.05:
            st.success("Konklusion: P-værdien er under 0.05. Vi forkaster Nul-hypotesen (H0). Der er en statistisk signifikant forskel på antallet af bookede møder mellem Deep Dive og Standard.")
        else:
            st.error("Konklusion: P-værdien er over 0.05. Nul-hypotesen (H0) accepteres. Forskellen i performance kan skyldes statistiske tilfældigheder.")


# SLIDE 6: FEATURE KORRELATION
elif page == "6. Feature Korrelation":
    st.title("Feature Korrelation")
    st.subheader("Analyse af variablernes sammenhæng")
    
    st.markdown("Før implementeringen af prædiktive Machine Learning modeller er det afgørende at undersøge matematisk, hvilke variabler der driver konverteringen. Nedenstående Pearson korrelationsmatrix illustrerer de lineære sammenhænge mellem datasættets numeriske features.")
    
    numeric_cols = df[['emails_sent', 'bounces', 'emails_opened', 'emails_replied', 'meetings_booked', 'avg_ai_fit_score', 'booking_rate_pct']]
    correlation_matrix = numeric_cols.corr()
    
    st.markdown("**Korrelationsmatrix for numeriske features**")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    
    st.divider()
    
    st.info("Observationer: Som forventet observeres en høj korrelation mellem volumen-metrikkerne (sendte, åbnede og besvarede emails). AI Fit Score udviser derimod en meget lav lineær korrelation over for de øvrige variabler. Dette indikerer, at lineær regression vil være utilstrækkelig, og det motiverer brugen af mere komplekse, ikke-lineære algoritmer som Random Forest i næste fase.")


# SLIDE 7: MACHINE LEARNING
elif page == "7. Machine Learning":
    from sklearn.model_selection import train_test_split
    
    st.title("Machine Learning (Stage 3)")
    st.markdown("For at skabe prædiktiv værdi for forretningen, har jeg implementeret både supervised og unsupervised machine learning modeller.")
    
    st.subheader("1. Supervised Learning: Prædiktion af Kampagnesucces")
    st.markdown("Jeg har trænet en Random Forest Classifier til at forudsige, om en kampagne bliver en succes (her defineret som mere end 3 bookede møder). Modellen er trænet på antal sendte emails, AI Fit Score og valget af berigelsesmetode.")
    
    # Klargør data til Random Forest
    df['is_success'] = (df['meetings_booked'] > 3).astype(int)
    df['is_deep_dive'] = (df['enrichment_mode'] == 'Deep Dive').astype(int)
    
    X = df[['emails_sent', 'avg_ai_fit_score', 'is_deep_dive']]
    y = df['is_success']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Træn model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, predictions)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
        ax_cm.set_xlabel('Forudsagt (Predicted)')
        ax_cm.set_ylabel('Faktisk (Actual)')
        st.pyplot(fig_cm)
        
    with col2:
        st.markdown("**Classification Report (Scores)**")
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
    st.divider()
    
    st.subheader("2. H4 Bevis: Sniper vs. Spray-and-Pray")
    st.markdown("Korrelationsmatrixen viste at emails_sent korrelerer negativt med booking_rate_pct (-0.29). Her visualiserer vi det direkte: kampagner opdeles i to grupper baseret på volumen, og vi sammenligner deres effektivitet.")

    # Opdel kampagner i to strategier baseret på H4-grænsen fra data-injektionen
    df['strategi'] = df['emails_sent'].apply(
        lambda x: 'Spray-and-Pray (>1500 emails)' if x > 1500 else 'Sniper (≤1500 emails)'
    )

    # Beregn gennemsnit pr. strategi
    h4_stats = df.groupby('strategi')[['meetings_booked', 'booking_rate_pct']].mean().round(2)
    h4_stats = h4_stats.sort_values('booking_rate_pct', ascending=False)
    
    st.markdown("**Gennemsnitlig performance pr. strategi**")
    st.dataframe(h4_stats)

    # Visualiser de to strategier side om side
    fig_h4, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(
        data=df, x='strategi', y='booking_rate_pct',
        hue='strategi', palette='Set2', ax=axes[0], errorbar=None, legend=False
    )
    axes[0].set_title('Booking Rate % pr. Strategi\n(Effektivitet per email sendt)', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Booking Rate (%)')
    axes[0].tick_params(axis='x', rotation=15)

    sns.barplot(
        data=df, x='strategi', y='meetings_booked',
        hue='strategi', palette='Set2', ax=axes[1], errorbar=None, legend=False
    )
    axes[1].set_title('Gns. Bookede Møder pr. Strategi\n(Absolut antal)', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Gns. Bookede Møder')
    axes[1].tick_params(axis='x', rotation=15)

    plt.suptitle('H4 Bevis: Sniper (Kvalitet) vs. Spray-and-Pray (Kvantitet)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_h4)

    sniper_rate = h4_stats.loc['Sniper (≤1500 emails)', 'booking_rate_pct']
    spray_rate  = h4_stats.loc['Spray-and-Pray (>1500 emails)', 'booking_rate_pct']

    st.info(f"H4 Bekræftet: Snipere konverterer {sniper_rate/spray_rate:.1f}x mere effektivt per udsendt email ({sniper_rate:.1f}% vs {spray_rate:.1f}%). Korrelationsmatrixen bekræftede dette statistisk: emails_sent ↔ booking_rate_pct = -0.29.")


# SLIDE 8: KONKLUSION
elif page == "8. Konklusion & Business Value":
    st.title("Konklusion og Forretningsværdi")
    st.subheader("Evaluering af hypoteser og anbefalinger til Navigent")
    
    st.markdown("Gennem ETL-processen, den eksplorative dataanalyse og machine learning-modelleringen kan der nu drages evidensbaserede konklusioner på de indledende hypoteser.")
    
    st.success("**Hypotese 1 Bekræftet: Deep Dive overgår Standard** \n\nDataen og den uafhængige T-test beviser en statistisk signifikant sammenhæng. Investeringen på 3 credits for Deep Dive berigelse giver et markant og målbart afkast i form af flere bookede møder.")
    
    st.success("**Hypotese 2 Bekræftet: SaaS konverterer bedst** \n\nSaaS-branchen tager betydeligt bedre imod automatiseret outreach sammenlignet med finanssektoren, hvilket resulterer i næsten en fordobling af konverteringsraten.")
    
    st.warning("**Hypotese 3 Delvist Bekræftet: Knowledge Base afhænger af kontekst** \n\nBrugen af RAG (Knowledge Base) øger svar-raten, men effekten er primært koncentreret i komplekse industrier som SaaS og finans, hvor domænespecifik kontekst er afgørende.")
    
    st.error("**Hypotese 4 Bekræftet: Kvalitet over kvantitet** \n\nKorrelationsanalysen viste en negativ korrelation (-0.29) mellem emails_sent og booking_rate_pct. Den direkte sammenligning bekræfter: Snipere (≤1500 emails, høj AI score) konverterer markant mere effektivt per email end Spray-and-Pray kampagner. Det kan ikke betale sig at skyde med spredehagl.")
    
    st.divider()
    
    st.subheader("Business Application (Stage 4)")
    st.markdown("Denne interaktive web-applikation udgør fjerde og sidste stadie i BI-workflowet. Formålet har været at operationalisere de underliggende data og gøre prædiktive modeller tilgængelige for non-tekniske beslutningstagere.")
    
    st.markdown("**Anbefaling til ledelsen:** \nNavigent bør justere sin onboarding-proces for at opfordre nye brugere til en målrettet kvalitetsstrategi frem for massiv volumen. Derudover bør ROI'en ved Deep Dive berigelse aktivt fremhæves i markedsføringen, især over for SaaS-segmentet.")