import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time

# Configuration de la page
st.set_page_config(
    page_title="Youbank - Simulation de Prêt Immobilier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Charger modèle, scaler et colonnes
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("model_Youbank_2.pkl")
    scaler = joblib.load("scaler_Youbank_2.pkl")
    feature_order = joblib.load("features_Youbank_2.pkl")
    return model, scaler, feature_order

try:
    model, scaler, feature_order = load_models()
except:
    st.error("Erreur lors du chargement des modèles. Vérifiez que les fichiers sont présents.")
    st.stop()

# -----------------------------
# Fonction pour calculer les features dérivées
# -----------------------------
def calculate_derived_features(df):
    df = df.copy()
    df['TotalIncome'] = df['ApplicantIncome_capped'] + df['CoapplicantIncome_capped']
    df['Dependents'] = df[['Dependents_1', 'Dependents_2', 'Dependents_3', 'Dependents_3+']].sum(axis=1)
    df['Charge_totale'] = df['Dependents'] * 500
    df['Income_to_Charge'] = df['TotalIncome'] / (1 + df['Charge_totale'])
    df['Rural'] = ((df['Property_Area_Semiurban'] == 0) & (df['Property_Area_Urban'] == 0)).astype(int)
    return df

# -----------------------------
# CSS personnalisé
# -----------------------------
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        border-bottom: 2px solid #1E40AF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #1E40AF;
        color: white;
        font-size: 1.2rem;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .accepted {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .rejected {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1E40AF;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# -----------------------------
# Page d'accueil
# -----------------------------
def show_home():
    st.markdown('<h1 class="main-header">🏦 Youbank - Simulation de Prêt Immobilier</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 1.3rem;'>
            Bienvenue sur notre simulateur de prêt immobilier. Obtenez une estimation instantanée 
            de votre éligibilité à un prêt immobilier en quelques clics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Commencer la simulation", key="start_btn"):
            st.session_state.page = "simulation"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
            <h3>📋 Informations nécessaires</h3>
            <ul>
            <li>Revenus mensuels</li>
            <li>Montant du prêt souhaité</li>
            <li>Durée de remboursement</li>
            <li>Historique de crédit</li>
            <li>Situation familiale et professionnelle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Page de simulation
# -----------------------------
def show_simulation():
    st.markdown('<h1 class="main-header">📋 Simulation de Prêt Immobilier</h1>', unsafe_allow_html=True)
    
    # Instructions
    st.info("Remplissez le formulaire ci-dessous avec vos informations. Tous les champs sont obligatoires.")
    
    # Organisation des champs en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">💵 Informations financières</div>', unsafe_allow_html=True)
        
        applicant_income = st.number_input(
            "Revenu mensuel du demandeur (€)", 
            min_value=0.0, 
            step=100.0,
            help="Votre revenu net mensuel"
        )
        
        coapplicant_income = st.number_input(
            "Revenu mensuel du co-demandeur (€)", 
            min_value=0.0, 
            step=100.0,
            help="Revenu net mensuel de votre conjoint(e) si applicable"
        )
        
        loan_amount = st.number_input(
            "Montant du prêt souhaité (€)", 
            min_value=0.0, 
            step=1000.0,
            help="Montant total que vous souhaitez emprunter"
        )
        
        loan_term = st.selectbox(
            "Durée du prêt (années)", 
            options=[7, 10, 15, 20, 25, 30],
            help="Durée de remboursement du prêt"
        )
        
        credit_history = st.selectbox(
            "Historique de crédit", 
            options=[("Bon historique", 1), ("Problèmes de crédit", 0)],
            format_func=lambda x: x[0],
            help="Avez-vous un bon historique de remboursement de crédit?"
        )
    
    with col2:
        st.markdown('<div class="sub-header">👤 Informations personnelles</div>', unsafe_allow_html=True)
        
        gender = st.selectbox(
            "Genre", 
            options=[("Femme", 0), ("Homme", 1)],
            format_func=lambda x: x[0]
        )
        
        married = st.selectbox(
            "Situation matrimoniale", 
            options=[("Célibataire", 0), ("Marié(e)", 1)],
            format_func=lambda x: x[0]
        )
        
        education = st.selectbox(
            "Niveau d'éducation", 
            options=[("Diplômé", 0), ("Non diplômé", 1)],
            format_func=lambda x: x[0]
        )
        
        self_employed = st.selectbox(
            "Statut professionnel", 
            options=[("Salarié", 0), ("Indépendant", 1)],
            format_func=lambda x: x[0]
        )
        
        property_area = st.selectbox(
            "Localisation du bien", 
            options=[("Rurale", 0), ("Semi-urbaine", 1), ("Urbaine", 2)],
            format_func=lambda x: x[0]
        )
        
        dependents = st.selectbox(
            "Personnes à charge", 
            options=[("Aucune", 0), ("1", 1), ("2", 2), ("3", 3), ("3+", 4)],
            format_func=lambda x: x[0]
        )
    
    # Conversion des sélections en valeurs numériques
    credit_history = credit_history[1]
    gender = 1 if gender[1] == 1 else 0
    married = married[1]
    education = education[1]
    self_employed = self_employed[1]
    
    # Conversion de la localisation en variables one-hot
    property_area_semiurban = 1 if property_area[1] == 1 else 0
    property_area_urban = 1 if property_area[1] == 2 else 0
    
    # Conversion des personnes à charge en variables one-hot
    dependents_1 = 1 if dependents[1] == 1 else 0
    dependents_2 = 1 if dependents[1] == 2 else 0
    dependents_3 = 1 if dependents[1] == 3 else 0
    dependents_3_plus = 1 if dependents[1] == 4 else 0
    
    # Conversion de la durée en mois
    loan_term_months = loan_term * 12
    
    # Préparation des données pour le modèle
    input_data = {
        "ApplicantIncome_capped": applicant_income,
        "CoapplicantIncome_capped": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term_months,
        "Credit_History": credit_history,
        "Gender_Male": gender,
        "Married_Yes": married,
        "Education_Not Graduate": education,
        "Self_Employed_Yes": self_employed,
        "Property_Area_Semiurban": property_area_semiurban,
        "Property_Area_Urban": property_area_urban,
        "Dependents_1": dependents_1,
        "Dependents_2": dependents_2,
        "Dependents_3": dependents_3,
        "Dependents_3+": dependents_3_plus
    }
    
    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Obtenir ma simulation", use_container_width=True):
            with st.spinner("Analyse de votre dossier en cours..."):
                time.sleep(1.5)  # Simuler un temps de traitement
                
                # Conversion en DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ajouter colonnes calculées
                input_df = calculate_derived_features(input_df)
                
                # Réorganiser selon l'ordre du modèle
                for col in feature_order:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_order]
                
                # Prédiction
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0, 1]
                
                # Affichage du résultat
                st.markdown("---")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box accepted">
                        <h2>✅ Félicitations ! Votre prêt est pré-approuvé</h2>
                        <p style="font-size: 1.2rem;">
                        Sur la base des informations fournies, vous avez de fortes chances d'obtenir votre prêt immobilier.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box rejected">
                        <h2>❌ Malheureusement, votre profil ne remplit pas nos critères</h2>
                        <p style="font-size: 1.2rem;">
                        Sur la base des informations fournies, nous ne pouvons pas approuver votre demande de prêt dans les conditions actuelles.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Barre de probabilité
                st.markdown("### Probabilité d'acceptation")
                st.progress(float(proba))
                st.markdown(f"**{proba*100:.1f}%** de chances d'obtention du prêt")
                
                # Recommandations
                st.markdown("---")
                st.markdown("### 📊 Analyse de votre dossier")
                
                if prediction == 0:
                    st.warning("""
                    **Pour améliorer vos chances d'obtention :**
                    - Augmentez votre apport personnel
                    - Réduisez le montant du prêt demandé
                    - Améliorez votre score de crédit
                    - Optez pour une durée de remboursement plus longue
                    """)
                
                st.success("""
                **Nos conseillers sont à votre disposition**
                Prenez rendez-vous avec l'un de nos conseillers financiers pour une étude plus détaillée de votre projet.
                """)
    
    # Bouton retour
    if st.button("← Retour à l'accueil", key="back_btn"):
        st.session_state.page = "home"
        st.rerun()

# -----------------------------
# Gestion de la navigation
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "simulation":
    show_simulation()