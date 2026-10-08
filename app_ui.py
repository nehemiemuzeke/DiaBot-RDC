import streamlit as st
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_URL = "http://localhost:8000/api/v1"

# ⚠️ REMPLACEZ "/ia/chat" PAR LE VRAI CHEMIN DE VOTRE API DE CONVERSATION DANS SWAGGER
# (regardez dans l'interface Swagger comment s'appelle le endpoint pour discuter)
CHAT_ENDPOINT = f"{API_URL}/ia/chat" 

st.set_page_config(page_title="DiaBot-RDC Chat", page_icon="🏥", layout="centered")

# ==============================================================================
# INITIALISATION DE L'ÉTAT (SESSION)
# ==============================================================================
if "token" not in st.session_state:
    st.session_state.token = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Mbote ! 👋 Je suis DiaBot-RDC.\n\nVeuillez vous **connecter** dans le menu de gauche pour commencer à discuter avec moi !"}
    ]

# ==============================================================================
# BARRE LATÉRALE (AUTHENTIFICATION)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100)
    st.title("🏥 DiaBot-RDC")
    
    if st.session_state.token is None:
        st.markdown("### 🔐 Connexion Patient")
        username = st.text_input("Nom d'utilisateur", value="patient1")
        password = st.text_input("Mot de passe", type="password", value="Password123!")
        
        if st.button("Se connecter", use_container_width=True):
            with st.spinner("Connexion en cours..."):
                try:
                    # Appel à votre API FastAPI pour le login
                    res = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
                    
                    if res.status_code in [200, 201]:
                        st.session_state.token = res.json().get("access_token")
                        st.success("✅ Connecté !")
                        st.rerun()  # Recharge l'interface
                    else:
                        st.error(f"Identifiants incorrects (Code {res.status_code})")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de joindre le serveur. Votre fichier main.py est-il lancé ?")
    else:
        st.success("✅ Vous êtes connecté au système sécurisé.")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.rerun()

# ==============================================================================
# FENÊTRE DE CHAT (STYLE CHATGPT)
# ==============================================================================
st.title("🧠 Conversation avec DiaBot")

# Afficher l'historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie du message
if prompt := st.chat_input("Ex: J'ai mangé du fufu avec du pondu, est-ce bon pour mon diabète ?"):
    
    if not st.session_state.token:
        st.error("⚠️ Vous devez vous connecter à gauche d'abord !")
    else:
        # 1. Afficher le message de l'utilisateur immédiatement
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Afficher la réponse de l'IA (avec animation de chargement)
        with st.chat_message("assistant"):
            with st.spinner("DiaBot réfléchit... 🤔"):
                
                # Préparation de la requête pour votre backend FastAPI
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                payload = {
                    "message": prompt,
                    "langue": "fr" # Peut être "ln", "sw", "kg", "lua"
                }

                try:
                    response = requests.post(CHAT_ENDPOINT, json=payload, headers=headers)
                    
                    if response.status_code in [200, 201]:
                        data = response.json()
                        
                        # ⚠️ IMPORTANT : Modifiez "texte" par la clé exacte renvoyée par votre API
                        # (Souvent c'est "texte", "reponse" ou "message")
                        reponse_ia = data.get("texte", "Désolé, je n'ai pas compris le format de retour.")
                        
                        st.markdown(reponse_ia)
                        st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
                    else:
                        st.error(f"Erreur API : {response.text}")
                except Exception as e:
                    st.error(f"Erreur de communication : {e}")