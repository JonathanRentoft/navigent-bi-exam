# Navigent BI Analysis: Cracking the SaaS Conversion Code

**Project:** Business Intelligence Exam Project (4. Semester)  
**Author:** Jonathan Rentoft Larsen, gruppe 8 

## Hvad går projektet ud på? (Problem Statement)
Jeg driver til daglig SaaS-virksomheden Navigent – en platform til leadgen og automatiseret outreach. Vi opererer med et "No-Cure-No-Pay" kreditsystem, hvor brugerne kan vælge mellem standard leads (1 credit) eller AI-berigede "Deep Dive" leads (3 credits). De kan også uploade dokumenter til vores Knowledge Base (RAG) for at optimere deres AI-genererede emails.

Problemstillingen er at undersøge, om brugernes investering i de dyre features (Deep Dive og Knowledge Base) rent faktisk kan betale sig i form af en højere konverteringsrate og flere bookede møder. Dette projekt er eksperimentel research, hvor jeg har udtrukket og syntetiseret kampagnedata for at bygge prædiktive modeller baseret på platformens performance.

## Mine hypoteser
Inden dataanalysen opstillede jeg følgende hypoteser:
1. **Deep Dive performer bedre end Standard:** Brugere, der benytter Deep Dive berigelse, opnår en signifikant højere rate af bookede møder.
2. **Brancheforskelle (SaaS > Finans):** Platformen performer bedst inden for SaaS-segmentet frem for mere konservative brancher som finans.
3. **RAG-optimering:** Kampagner med Knowledge Base aktiveret konverterer bedre end dem uden (særligt inden for komplekse brancher).
4. **Volumen vs. Abonnement:** Antagelsen er, at brugere der sender flest mails får flest møder per 100 emails, uanset om de bruger deep dive eller standard søgninger.


## Data Preparation & Cleaning (Stage 2)
For at klargøre data til maskinlæring, blev der udført en grundig ETL-proces i Jupyter Notebook:
* **Kategoriseringsfejl:** Standardisering af inkonsistent tekst i `target_industry` (f.eks. samling af 'saas', 'SaaS' og ' SaaS ').
* **Missing values:** Imputering af manglende værdier i `plan_tier` med 'Unknown'.
* **Outliers:** Identificering og filtrering af ekstreme outliers (f.eks. fejlindtastede kampagner med over 10.000 emails) ved hjælp af den statistiske IQR-metode.
* **Datatyper:** Konvertering af tekst-fejl ("N/A") i numeriske kolonner til 0, hvorefter de blev castet til integers for at muliggøre beregninger.

## Data Modelling & Machine Learning (Stage 3)
Scikit-learn blev anvendt til at bygge prædiktive modeller og finde mønstre i dataen:
* **Supervised Learning (Random Forest Classifier):** Udvikling af en prædiktiv model til at forudsige kampagnesucces (>3 bookede møder) baseret på tier, feature-brug og branche. Modellen blev valideret via en Confusion Matrix samt F1, Precision og Recall scores.
* **Unsupervised Learning (K-Means Clustering):** Identifikation af tre tydelige kundesegmenter baseret på volumen og AI-kvalitet (Spammerne, Sniperne og Gennemsnittet). Data blev skaleret via StandardScaler forud for clustering.

## Konklusioner
Gennem eksplorativ dataanalyse og hypotesetest fremstod følgende resultater:
* **Hypotese 1 bekræftet:** Deep Dive giver betydeligt flere møder. En uafhængig T-test (p-værdi < 0.05) bekræftede en statistisk signifikant forskel mellem Deep Dive og Standard leads.
* **Hypotese 2 bekræftet:** SaaS-branchen tager markant bedre imod automatiseret outreach end finans. Konverteringsraten er næsten dobbelt så høj.
* **Hypotese 3 delvist bekræftet:** Knowledge Base giver et tydeligt boost i svar-raten, men primært for komplekse industrier som SaaS og finans.
* **Hypotese 4 afkræftet:** Clustering-analysen påviste, at antagelsen om volumen-drevet succes ('Spray-and-pray') er fejlagtig. De mest succesfulde brugere (Sniper-segmentet) udsender færre emails totalt set, men opretholder en høj AI Fit Score og lukker dermed flere møder pr. udsendt email.


## Business Application & Usability (Stage 4)
Løsningen er gjort tilgængelig som en interaktiv web-applikation bygget i Streamlit, der gør det muligt for non-tekniske interessenter at filtrere og visualisere dataen live.

**Usability Evaluation:** Inden aflevering blev prototypen præsenteret for en testperson. Baseret på feedback blev der implementeret en sidebar til industrifiltrering samt fremhævede KPI-kort i toppen af interfacet for at gøre det samlede overblik mere intuitivt og tilgængeligt.

**Sådan køres applikationen lokalt:**
1. Klon repository
2. Kør `pip install -r requirements.txt`
3. Start appen med `streamlit run app.py`
