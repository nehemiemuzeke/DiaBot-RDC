import streamlit as st
import requests
import os
import time

st.set_page_config(
    page_title="DiaBot-RDC",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANGUAGES = {
    "Français": "fr",
    "Lingala": "ln",
    "Swahili": "sw",
    "Kikongo": "kg",
    "Tshiluba": "lu",
}

SYSTEM_PROMPT = """Tu es DiaBot-RDC, un assistant médical intelligent, empathique et chaleureux spécialisé dans la gestion du diabète en République Démocratique du Congo (RDC).

DIRECTIVES IMPORTANTES :
1. Sois humain, naturel et amical. Si l'utilisateur se présente ou te salue (ex: "je m'appelle Néhémie"), salue-le personnellement et chaleureusement par son prénom.
2. Adapte tes conseils au contexte congolais (fufu, pondu, saka-saka, madesu, chikwangue, bananes plantains, poisson).
3. Donne des conseils clairs, scientifiquement justes mais accessibles à tous.
4. En cas d'urgence (hypoglycémie < 0.70 g/L), donne immédiatement la règle des 15g de sucre.
"""

SVG_LOGO = """<svg width="36" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#4285F4"/><stop offset="50%" stop-color="#9b72cb"/><stop offset="100%" stop-color="#d96570"/>
</linearGradient></defs>
<circle cx="16" cy="16" r="15" fill="url(#lg)"/>
<path d="M16 7l2.47 5.01L24 13.24l-4 3.89.94 5.51L16 19.77l-4.94 2.87.94-5.51-4-3.89 5.53-.83L16 7z" fill="white"/>
</svg>"""

SVG_STAR = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="sg" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#4285F4"/><stop offset="50%" stop-color="#9b72cb"/><stop offset="100%" stop-color="#d96570"/>
</linearGradient></defs>
<path d="M12 2l2.94 5.96L21 9.19l-4.5 4.38L17.56 20 12 16.9 6.44 20l1.06-6.43L3 9.19l6.06-1.23L12 2z" fill="url(#sg)"/>
</svg>"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0d10;
  --surface: #14161a;
  --surface-hover: #1e2026;
  --input: #1e1f23;
  --border: #2a2c31;
  --text: #e8eaed;
  --muted: #9aa0a6;
  --blue: #4285F4;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Masquer les éléments inutiles SANS masquer le bouton d'ouverture de la Sidebar */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
}

/* Rendre le header transparent pour laisser voir le bouton flèche de la sidebar */
[data-testid="stHeader"] {
  background-color: transparent !important;
  z-index: 100 !important;
}

/* Style du bouton d'ouverture/fermeture de la Sidebar */
[data-testid="stHeader"] button {
  color: var(--text) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 2rem !important;
  max-width: 100% !important;
}

[data-testid="stSidebar"] {
  background: #0f1115 !important;
  border-right: 1px solid var(--border) !important;
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0 20px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.sidebar-header .title {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, #4285F4, #9b72cb, #d96570);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sidebar-header .sub {
  font-size: 11px;
  color: var(--muted);
}

.sidebar-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.8px;
  color: var(--muted);
  text-transform: uppercase;
  margin: 16px 0 8px 0;
}

.welcome-container {
  min-height: 48vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  position: relative;
  padding: 40px 20px 20px 20px;
}

.halo {
  position: absolute;
  width: 480px;
  height: 480px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(66,133,244,0.18) 0%, rgba(155,114,203,0.08) 35%, transparent 70%);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
  animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.6; transform: translate(-50%, -50%) scale(1); }
  50% { opacity: 1; transform: translate(-50%, -50%) scale(1.1); }
}

.welcome-title {
  font-size: 36px;
  font-weight: 700;
  background: linear-gradient(135deg, #4285F4, #9b72cb, #d96570);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 12px;
  position: relative;
  z-index: 1;
}

.welcome-sub {
  font-size: 15px;
  color: var(--muted);
  max-width: 480px;
  position: relative;
  z-index: 1;
  line-height: 1.6;
}

.chat-wrap {
  max-width: 768px;
  margin: 0 auto;
  padding: 12px 16px 120px 16px;
}

.msg {
  display: flex;
  gap: 14px;
  margin-bottom: 28px;
  align-items: flex-start;
}

.msg-user {
  flex-direction: row-reverse;
}

.av {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
}

.av-ai {
  background: linear-gradient(135deg, #4285F4, #9b72cb);
}

.av-user {
  background: #2f2f2f;
  color: var(--text);
}

.bubble-user {
  max-width: 85%;
  background: #2f2f2f;
  color: var(--text);
  border-radius: 18px;
  padding: 12px 18px;
  font-size: 15px;
  line-height: 1.65;
}

.text-ai {
  color: var(--text);
  font-size: 15px;
  line-height: 1.75;
  padding-top: 4px;
}

.dots {
  display: flex;
  gap: 6px;
  padding: 10px 0;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #8ab4f8;
  animation: bounce 1.3s ease-in-out infinite;
}

.dot:nth-child(2) { animation-delay: 0.15s; }
.dot:nth-child(3) { animation-delay: 0.3s; }

@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
  40% { transform: translateY(-8px); opacity: 1; }
}

[data-testid="stChatInput"] {
  max-width: 768px !important;
  margin: 0 auto !important;
}

[data-testid="stChatInput"] > div {
  background: var(--input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 28px !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

[data-testid="stChatInput"] textarea {
  color: var(--text) !important;
  font-size: 15px !important;
}

.sugg .stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  padding: 12px 16px !important;
  height: auto !important;
  white-space: normal !important;
  text-align: left !important;
  box-shadow: none !important;
  transition: all 0.2s ease !important;
}

.sugg .stButton > button:hover {
  background: var(--surface-hover) !important;
  border-color: #8ab4f8 !important;
  transform: translateY(-2px) !important;
}

div[data-baseweb="select"] > div, .stTextInput input {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

.stButton > button {
  border-radius: 10px !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# State Management
defaults = {
    "messages": [],
    "username": "Patient",
    "langue": "fr",
    "pending_prompt": None,
    "groq_key": os.getenv("GROQ_API_KEY", ""),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def query_groq_api(messages_history, user_langue, api_key):
    """Inférence REST directe vers Groq avec gestion d'erreurs claire."""
    api_key = (api_key or "").strip()

    if not api_key:
        return (
            "⚠️ **Clé API Groq manquante !**\n\n"
            "Pour discuter avec l'IA LLaMA 3.3, collez votre clé gratuite (commençant par `gsk_...`) "
            "dans le champ **🔑 CLÉ API GROQ** dans le menu de gauche (Sidebar).\n\n"
            "👉 Obtenez une clé gratuitement sur [console.groq.com/keys](https://console.groq.com/keys)."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    lang_instruction = f"Réponds impérativement dans la langue : {user_langue}."
    formatted_messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n{lang_instruction}"}]

    for m in messages_history:
        formatted_messages.append({
            "role": "user" if m["role"] == "user" else "assistant",
            "content": m["content"]
        })

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]

    last_error = ""

    for model_name in models:
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=20)

            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
            elif res.status_code == 401:
                return (
                    "❌ **Clé API Groq invalide (Erreur 401)**.\n\n"
                    "Assurez-vous d'avoir copié la clé complète commençant par `gsk_` "
                    "depuis [console.groq.com/keys](https://console.groq.com/keys)."
                )
            else:
                last_error = f"Erreur {res.status_code}: {res.text}"
        except Exception as e:
            last_error = str(e)

    return f"⚠️ Problème de connexion avec Groq : {last_error}"


# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
            {SVG_LOGO}
            <div>
                <div class="title">DiaBot-RDC</div>
                <div class="sub">Assistant Diabète Intelligent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("➕ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown('<div class="sidebar-label">🌐 LANGUE</div>', unsafe_allow_html=True)
    lang = st.selectbox("lang_select", list(LANGUAGES.keys()), index=0, label_visibility="collapsed")
    st.session_state.langue = LANGUAGES[lang]

    st.markdown('<div class="sidebar-label">👤 VOTRE PRÉNOM</div>', unsafe_allow_html=True)
    custom_name = st.text_input("Name", value=st.session_state.username, label_visibility="collapsed")
    if custom_name:
        st.session_state.username = custom_name

    st.markdown('<div class="sidebar-label">🔑 CLÉ API GROQ</div>', unsafe_allow_html=True)
    input_key = st.text_input("Groq Key", value=st.session_state.groq_key, type="password", placeholder="gsk_...", label_visibility="collapsed")
    if input_key:
        st.session_state.groq_key = input_key

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    initial = (st.session_state.username or "P")[:1].upper()
    status_text = "● Clé configurée" if st.session_state.groq_key else "○ Entrez votre clé gsk_"
    status_color = "#34a853" if st.session_state.groq_key else "#e57373"

    st.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;padding:12px;border:1px solid #2a2c31;
             background:#14161a;border-radius:12px;">
          <div style="width:36px;height:36px;border-radius:50%;background:#2f2f2f;display:flex;
               align-items:center;justify-content:center;font-weight:600;font-size:14px;">{initial}</div>
          <div>
            <div style="font-size:13px;font-weight:600;">{st.session_state.username}</div>
            <div style="font-size:11px;color:{status_color};">{status_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Zone de Chat
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-container">
          <div class="halo"></div>
          <div class="welcome-title">Bonjour, comment puis-je vous aider ?</div>
          <div class="welcome-sub">
            Posez-moi vos questions sur le diabète, l'alimentation ou vos traitements en RDC.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Puis-je manger du fufu ?",
        "Symptômes d'hypoglycémie ?",
        "Comment surveiller ma glycémie ?",
        "Aliments recommandés ?",
    ]
    cols = st.columns(4)
    for i, s in enumerate(suggestions):
        with cols[i]:
            st.markdown('<div class="sugg">', unsafe_allow_html=True)
            if st.button(s, key=f"s_{i}", use_container_width=True):
                st.session_state.pending_prompt = s
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
else:
    parts = ['<div class="chat-wrap">']
    for m in st.session_state.messages:
        if m["role"] == "user":
            initial = (st.session_state.username or "P")[:1].upper()
            parts.append(
                f'<div class="msg msg-user"><div class="av av-user">{initial}</div>'
                f'<div class="bubble-user">{m["content"]}</div></div>'
            )
        else:
            parts.append(
                f'<div class="msg"><div class="av av-ai">{SVG_STAR}</div>'
                f'<div class="text-ai">{m["content"]}</div></div>'
            )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# Saisie Utilisateur
prompt = st.chat_input("Posez votre question sur le diabète…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    thinking_html = f"""
    <div class="chat-wrap">
      <div class="msg">
        <div class="av av-ai">{SVG_STAR}</div>
        <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
      </div>
    </div>
    """
    box = st.empty()
    box.markdown(thinking_html, unsafe_allow_html=True)

    # Inférence directe Groq REST API
    reply = query_groq_api(
        st.session_state.messages,
        st.session_state.langue,
        st.session_state.groq_key
    )

    # Effet de streaming
    base = ['<div class="chat-wrap">']
    for m in st.session_state.messages:
        if m["role"] == "user":
            initial = (st.session_state.username or "P")[:1].upper()
            base.append(
                f'<div class="msg msg-user"><div class="av av-user">{initial}</div>'
                f'<div class="bubble-user">{m["content"]}</div></div>'
            )
        else:
            base.append(
                f'<div class="msg"><div class="av av-ai">{SVG_STAR}</div>'
                f'<div class="text-ai">{m["content"]}</div></div>'
            )
    base_html = "".join(base)

    acc = ""
    for i, ch in enumerate(reply):
        acc += ch
        if i % 4 == 0 or i == len(reply) - 1:
            box.markdown(
                base_html
                + f'<div class="msg"><div class="av av-ai">{SVG_STAR}</div>'
                + f'<div class="text-ai">{acc}</div></div></div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.003)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()