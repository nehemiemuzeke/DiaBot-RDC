import streamlit as st
import requests
import time

st.set_page_config(
    page_title="DiaBot-RDC",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Base URL Render
RENDER_BASE_URL = "https://diabot-api.onrender.com"

LANGUAGES = {
    "Français": "fr",
    "Lingala": "ln",
    "Swahili": "sw",
    "Kikongo": "kg",
    "Tshiluba": "lu",
}

SVG_LOGO = """<svg width="48" height="48" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#4285F4"/><stop offset="50%" stop-color="#9b72cb"/><stop offset="100%" stop-color="#d96570"/>
</linearGradient></defs>
<circle cx="16" cy="16" r="15" fill="url(#lg)"/>
<path d="M16 7l2.47 5.01L24 13.24l-4 3.89.94 5.51L16 19.77l-4.94 2.87.94-5.51-4-3.89 5.53-.83L16 7z" fill="white"/>
</svg>"""

SVG_STAR = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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

#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 1rem !important;
  max-width: 100% !important;
}

[data-testid="stSidebar"] {
  background: #0e1014 !important;
  border-right: 1px solid var(--border) !important;
}

/* Splash */
.splash {
  position: fixed; inset: 0; z-index: 99999;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  background: var(--bg);
  animation: splashHide 0.7s ease-in-out 2.1s forwards;
  pointer-events: none;
}
.splash-halo {
  position: absolute; width: 420px; height: 420px; border-radius: 50%;
  background: radial-gradient(circle, rgba(66,133,244,.22) 0%, rgba(155,114,203,.08) 40%, transparent 70%);
  animation: pulse 2.4s ease-in-out infinite;
}
.splash-title {
  margin-top: 16px; font-size: 40px; font-weight: 700; letter-spacing: -1px;
  background: linear-gradient(135deg, #4285F4, #9b72cb, #d96570);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: fadeUp .7s ease-out .25s both;
}
.splash-sub {
  margin-top: 8px; color: var(--muted); font-size: 14px;
  animation: fadeUp .7s ease-out .4s both;
}
.splash-bar {
  width: 110px; height: 3px; margin-top: 26px; border-radius: 2px;
  background: linear-gradient(90deg, #4285F4, #9b72cb, #d96570, #4285F4);
  background-size: 200% 100%;
  animation: shimmer 1.4s linear infinite, fadeUp .7s ease-out .55s both;
}
@keyframes splashHide { to { opacity: 0; visibility: hidden; } }
@keyframes pulse {
  0%,100% { transform: scale(1); opacity: .55; }
  50% { transform: scale(1.12); opacity: 1; }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}

/* Login */
.login-shell {
  min-height: 85vh; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  position: relative; padding: 24px 12px;
}
.login-halo {
  position: absolute; width: 480px; height: 480px; border-radius: 50%;
  background: radial-gradient(circle, rgba(66,133,244,.12) 0%, rgba(155,114,203,.05) 40%, transparent 70%);
  top: 45%; left: 50%; transform: translate(-50%, -50%);
  animation: pulse 4s ease-in-out infinite; pointer-events: none;
}
.brand { text-align: center; position: relative; z-index: 2; margin-bottom: 22px; }
.brand h1 {
  margin: 12px 0 6px; font-size: 28px; font-weight: 700;
  background: linear-gradient(135deg, #4285F4, #9b72cb, #d96570);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand p { color: var(--muted); font-size: 13px; margin: 0; }

.login-card {
  width: 100%; max-width: 380px; position: relative; z-index: 2;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 28px 24px 22px;
  box-shadow: 0 10px 40px rgba(0,0,0,.45), 0 0 40px rgba(66,133,244,.12);
}
.login-card h2 {
  margin: 0 0 18px; text-align: center; font-size: 16px; font-weight: 600; color: var(--text);
}

div[data-testid="stTextInput"] label {
  color: var(--muted) !important; font-size: 12px !important; font-weight: 500 !important;
}
div[data-testid="stTextInput"] input {
  background: var(--input) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important;
  height: 46px !important; font-size: 14px !important; padding: 0 14px !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 2px rgba(66,133,244,.2) !important;
}

.login-btn .stButton > button {
  width: 100% !important; height: 46px !important; border: none !important;
  border-radius: 12px !important; margin-top: 6px !important;
  background: linear-gradient(135deg, #4285F4 0%, #1a73e8 100%) !important;
  color: white !important; font-weight: 600 !important; font-size: 14px !important;
  box-shadow: 0 4px 16px rgba(66,133,244,.3) !important;
}
.login-btn .stButton > button:hover {
  background: linear-gradient(135deg, #5a95f5 0%, #2b7de9 100%) !important;
  transform: translateY(-1px) !important;
}

.foot {
  margin-top: 18px; color: #5f6368; font-size: 11px; text-align: center;
  position: relative; z-index: 2;
}

/* Chat */
.chat-wrap { max-width: 768px; margin: 0 auto; padding: 12px 16px 110px; }
.msg { display: flex; gap: 12px; margin-bottom: 24px; align-items: flex-start; animation: fadeUp .3s ease-out; }
.msg-user { flex-direction: row-reverse; }
.av {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600;
}
.av-ai { background: linear-gradient(135deg, #4285F4, #9b72cb); }
.av-user { background: #2f2f2f; color: var(--text); }
.bubble-user {
  max-width: 85%; background: #2f2f2f; color: var(--text);
  border-radius: 18px; padding: 12px 16px; font-size: 15px; line-height: 1.6;
}
.text-ai { color: var(--text); font-size: 15px; line-height: 1.75; padding-top: 4px; }

.welcome {
  min-height: 58vh; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center; position: relative;
}
.welcome h1 {
  position: relative; z-index: 1; font-size: 34px; font-weight: 700; margin: 0 0 10px;
  background: linear-gradient(135deg, #4285F4, #9b72cb, #d96570);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.welcome p { position: relative; z-index: 1; color: var(--muted); font-size: 15px; max-width: 420px; }

.dots { display: flex; gap: 5px; padding: 8px 0; }
.dot {
  width: 7px; height: 7px; border-radius: 50%; background: #8ab4f8;
  animation: bounce 1.3s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: .15s; }
.dot:nth-child(3) { animation-delay: .3s; }
@keyframes bounce {
  0%,80%,100% { transform: translateY(0); opacity: .35; }
  40% { transform: translateY(-8px); opacity: 1; }
}

[data-testid="stChatInput"] { max-width: 768px !important; margin: 0 auto !important; }
[data-testid="stChatInput"] > div {
  background: var(--input) !important; border: 1px solid var(--border) !important;
  border-radius: 28px !important; box-shadow: 0 2px 16px rgba(0,0,0,.3) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-size: 15px !important; }

.sugg .stButton > button {
  background: #1a1c20 !important; border: 1px solid var(--border) !important;
  border-radius: 22px !important; color: var(--text) !important;
  font-size: 13px !important; font-weight: 400 !important;
  padding: 12px 14px !important; height: auto !important;
  white-space: normal !important; text-align: left !important;
  box-shadow: none !important;
}
.sugg .stButton > button:hover {
  border-color: #8ab4f8 !important; transform: translateY(-2px) !important;
}

div[data-baseweb="select"] > div {
  background: #1a1c20 !important; border-color: var(--border) !important;
  border-radius: 10px !important; color: var(--text) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# State
defaults = {
    "token": None,
    "messages": [],
    "username": "",
    "langue": "fr",
    "splash_done": False,
    "pending_prompt": None,
    "working_prefix": "/api/v1",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def login(username: str, password: str):
    # Teste automatiquement /api/v1/auth/login ET /auth/login
    prefixes = ["/api/v1", ""]
    base = RENDER_BASE_URL.rstrip("/")

    for prefix in prefixes:
        url = f"{base}{prefix}/auth/login"
        try:
            # Essai JSON
            r = requests.post(url, json={"username": username, "password": password}, timeout=25)
            if r.status_code in (200, 201):
                data = r.json()
                token = data.get("access_token") or data.get("token")
                if token:
                    st.session_state.working_prefix = prefix
                    return True, token, None
            elif r.status_code in (400, 401, 422):
                return False, None, "Nom d'utilisateur ou mot de passe incorrect."

            # Essai Form-Data
            r = requests.post(url, data={"username": username, "password": password}, timeout=25)
            if r.status_code in (200, 201):
                data = r.json()
                token = data.get("access_token") or data.get("token")
                if token:
                    st.session_state.working_prefix = prefix
                    return True, token, None
        except requests.exceptions.Timeout:
            return False, None, "Le serveur Render se réveille encore. Veuillez recliquer sur Se connecter dans 10 secondes !"
        except requests.exceptions.ConnectionError:
            return False, None, "Impossible de joindre le serveur. Vérifiez l'URL Render."
        except Exception:
            pass

    return False, None, "Erreur API : Impossible de trouver la route de connexion."


# Splash
if not st.session_state.splash_done and not st.session_state.token:
    st.markdown(
        f"""
        <div class="splash">
          <div class="splash-halo"></div>
          <div>{SVG_LOGO}</div>
          <div class="splash-title">DiaBot-RDC</div>
          <div class="splash-sub">Assistant médical intelligent</div>
          <div class="splash-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(2.4)
    st.session_state.splash_done = True
    st.rerun()


# Sidebar
if st.session_state.token:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0 14px;">
              {SVG_LOGO.replace('width="48"', 'width="28"').replace('height="48"', 'height="28"')}
              <span style="font-weight:700;font-size:16px;background:linear-gradient(135deg,#4285F4,#9b72cb,#d96570);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">DiaBot-RDC</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("LANGUE")
        lang = st.selectbox("lang", list(LANGUAGES.keys()), label_visibility="collapsed")
        st.session_state.langue = LANGUAGES[lang]

        if st.button("Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        initial = (st.session_state.username or "U")[:1].upper()
        st.markdown(
            f"""
            <div style="display:flex;gap:10px;align-items:center;padding:10px;border:1px solid #2a2c31;
                 background:#1a1c20;border-radius:10px;margin-bottom:8px;">
              <div style="width:34px;height:34px;border-radius:50%;background:#2f2f2f;display:flex;
                   align-items:center;justify-content:center;font-weight:600;">{initial}</div>
              <div>
                <div style="font-size:13px;font-weight:500;">{st.session_state.username}</div>
                <div style="font-size:11px;color:#34a853;">En ligne</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Déconnexion", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.username = ""
            st.session_state.pending_prompt = None
            st.rerun()


# LOGIN
if not st.session_state.token:
    st.markdown(
        f"""
        <div class="login-shell">
          <div class="login-halo"></div>
          <div class="brand">
            {SVG_LOGO}
            <h1>DiaBot-RDC</h1>
            <p>Assistant intelligent pour la gestion du diabète</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.15, 1])
    with c2:
        st.markdown('<div class="login-card"><h2>Connexion Patient</h2>', unsafe_allow_html=True)
        username = st.text_input("Nom d'utilisateur", value="patient1", key="login_user")
        password = st.text_input("Mot de passe", value="Password123!", type="password", key="login_pass")
        st.markdown('<div class="login-btn">', unsafe_allow_html=True)
        submit = st.button("Se connecter", use_container_width=True, key="login_submit")
        st.markdown("</div></div>", unsafe_allow_html=True)

        if submit:
            if not username or not password:
                st.error("Veuillez remplir tous les champs.")
            else:
                with st.spinner("Connexion en cours..."):
                    ok, token, err = login(username.strip(), password)
                if ok:
                    st.session_state.token = token
                    st.session_state.username = username.strip()
                    st.session_state.messages = []
                    st.session_state.pending_prompt = None
                    st.success("Connexion réussie")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error(err or "Échec de connexion")

    st.markdown('<div class="foot">DiaBot-RDC · Données médicales sécurisées</div>', unsafe_allow_html=True)


# CHAT
else:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              <div class="login-halo"></div>
              <h1>Bonjour, comment puis-je vous aider ?</h1>
              <p>Posez vos questions sur le diabète, l'alimentation ou les traitements.</p>
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
                initial = (st.session_state.username or "U")[:1].upper()
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

    prompt = st.chat_input("Posez votre question sur le diabète…")
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        thinking = f"""
        <div class="chat-wrap">
          <div class="msg">
            <div class="av av-ai">{SVG_STAR}</div>
            <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
          </div>
        </div>
        """
        box = st.empty()
        box.markdown(thinking, unsafe_allow_html=True)

        reply = "Je n'ai pas pu obtenir de réponse."
        chat_url = f"{RENDER_BASE_URL.rstrip('/')}{st.session_state.working_prefix}/ia/chat"

        try:
            res = requests.post(
                chat_url,
                json={"message": prompt, "langue": st.session_state.langue},
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                timeout=45,
            )
            if res.status_code in (200, 201):
                data = res.json()
                reply = (
                    data.get("texte")
                    or data.get("reponse")
                    or data.get("message")
                    or data.get("response")
                    or reply
                )
            elif res.status_code == 401:
                st.session_state.token = None
                st.warning("Session expirée. Reconnectez-vous.")
                st.rerun()
            else:
                reply = f"Erreur API ({res.status_code})."
        except requests.exceptions.Timeout:
            reply = "Le serveur a mis trop de temps à répondre. Réessayez votre question."
        except requests.exceptions.ConnectionError:
            reply = "Impossible de joindre l'API sur Render."
        except Exception as e:
            reply = f"Erreur: {e}"

        base = ['<div class="chat-wrap">']
        for m in st.session_state.messages:
            if m["role"] == "user":
                initial = (st.session_state.username or "U")[:1].upper()
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
                time.sleep(0.005)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()