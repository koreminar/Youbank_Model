import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Charger modèle, scaler et colonnes
# -----------------------------
model = joblib.load("model_Youbank_2.pkl")
scaler = joblib.load("scaler_Youbank_2.pkl")
feature_order = joblib.load("features_Youbank_2.pkl")

# -----------------------------
# Colonnes saisies par utilisateur
# -----------------------------
user_input_features = [
    "ApplicantIncome_capped", "CoapplicantIncome_capped", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Gender_Male", "Married_Yes", "Education_Not Graduate",
    "Self_Employed_Yes", "Property_Area_Semiurban", "Property_Area_Urban",
    "Dependents_1", "Dependents_2", "Dependents_3", "Dependents_3+"
]

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
# Interface Streamlit
# -----------------------------
st.title("🏦 Prédiction Prêt Immobilier - Youbank 2")

st.markdown("Entrez les informations du dossier :")

input_data = {}
for feature in user_input_features:
    if feature in [
        "Credit_History", "Gender_Male", "Married_Yes", "Education_Not Graduate",
        "Self_Employed_Yes", "Property_Area_Semiurban", "Property_Area_Urban",
        "Dependents_1", "Dependents_2", "Dependents_3", "Dependents_3+"
    ]:
        input_data[feature] = st.selectbox(f"{feature.replace('_', ' ')}", [0, 1])
    elif feature == "Loan_Amount_Term":
        # Nouvelle saisie en mois
        input_data[feature] = st.selectbox("Durée du prêt (mois)", [90,180, 360])
    else:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ')}", min_value=0.0, step=100.0)

input_df = pd.DataFrame([input_data])

# Ajouter colonnes calculées
input_df = calculate_derived_features(input_df)

# Réorganiser selon l’ordre du modèle
for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_order]

# -----------------------------
# Prédiction
# -----------------------------
if st.button("🔮 Prédire le résultat du prêt"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0, 1]

    if prediction == 1:
        st.success(f"✅ Prêt **Accepté**")
    else:
        st.error(f"❌ Prêt **Refusé**")
    
    # Barre de probabilité
    st.markdown("### Probabilité d'acceptation du prêt")
    st.progress(int(proba * 100))
    st.write(f"{proba:.2%}")
