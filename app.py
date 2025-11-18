import streamlit as st
from models import process_prompt
from datetime import datetime
import os
import re

# === Restored API key as requested (user provided) ===
os.environ["OPEN_ROUTER"] = st.secrets["OPEN_ROUTER"]

# --- Try optional spellchecker ---
try:
    from spellchecker import SpellChecker
    SPELL_AVAILABLE = True
    spell = SpellChecker()
except Exception:
    SPELL_AVAILABLE = False
    spell = None

# --- Constants ---
DARK_BG_COLOR = "#aeaeae"
DARK_TEXT_COLOR = "#ffffff"
INPUT_BG_COLOR = "#003366"
INPUT_TEXT_COLOR = "white"
PRIMARY_COLOR = "#003366"

# --- Page setup ---
st.set_page_config(page_title="Prompt Classifier Chatbot", page_icon="🤖", layout="centered")

# =========================
# Team Section (Sidebar)
# =========================
st.sidebar.title("👥 Project Team")

team_members = {
    "Rana": "https://github.com/RanaSaleh2",
    "Haitham": "https://github.com/Ha80ii",
    "Danah": "https://github.com/dralsarrani",
    "Ghadi": "https://github.com/GhadiBa",
}

for name, url in team_members.items():
    st.sidebar.markdown(f"- [{name}]({url})")
st.markdown("---")

# --- Session state ---
ms = st.session_state
if "messages" not in ms:
    ms.messages = []

if "themes" not in ms:
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "light",
            "theme.backgroundColor": "#FFFFFF",
            "theme.primaryColor": PRIMARY_COLOR,
            "theme.secondaryBackgroundColor": "#F1F0F0",
            "theme.textColor": "#000000",
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": DARK_BG_COLOR,
            "theme.primaryColor": PRIMARY_COLOR,
            "theme.secondaryBackgroundColor": PRIMARY_COLOR,
            "theme.textColor": DARK_TEXT_COLOR,
        },
    }

# --- Helper functions ---

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def detect_safety_tag(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    unsafe_keys = ["unsafe", "mismatch", "review", "block", "danger", "not allowed", "forbidden"]
    for k in unsafe_keys:
        if k in t:
            return "unsafe"
    safe_keys = ["safe", "allowed", "approved"]
    for k in safe_keys:
        if k in t and "unsafe" not in t:
            return "safe"
    return ""

def safety_emoji(tag: str) -> str:
    if tag == "safe":
        return " 🛡"
    if tag == "unsafe":
        return " ⚠"
    return ""

def simple_spell_suggest(text: str) -> str:
    if not SPELL_AVAILABLE or not spell:
        return text
    parts = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
    corrected_parts = []
    for p in parts:
        if re.match(r"^\w+$", p):
            w = p
            if w.isdigit():
                corrected_parts.append(w)
                continue
            if w.lower() in spell:
                corrected_parts.append(p)
            else:
                corr = spell.correction(w)
                if corr and corr.lower() != w.lower():
                    if w.istitle():
                        corr = corr.title()
                    elif w.isupper():
                        corr = corr.upper()
                    corrected_parts.append(corr)
                else:
                    corrected_parts.append(p)
        else:
            corrected_parts.append(p)
    return "".join(corrected_parts)

def format_message(msg, role):
    time_str = msg["time"].strftime("%H:%M")
    if role == "user":
        return f"""
        <div style="display:flex; justify-content:flex-end; margin:5px 0;">
            <div style="background-color:#ADD8E6; color:black; padding:10px; border-radius:10px; max-width:80%; display:flex; align-items:center; direction:ltr;">
                <span style="color:{PRIMARY_COLOR}; margin-right:5px;">👤</span>
                <span>{msg['content']}</span>
                <span style="font-size:0.7em;color:#555; margin-left:5px;">[{time_str}]</span>
            </div>
        </div>
        """
    else:
        tag = detect_safety_tag(msg.get("content",""))
        emoji = safety_emoji(tag)
        return f"""
        <div style="display:flex; justify-content:flex-start; margin:5px 0;">
            <div style="background-color:{PRIMARY_COLOR}; color:white; padding:10px; border-radius:10px; max-width:80%; display:flex; align-items:center;">
                <span style="color:white; margin-right:5px;">🤖{emoji}</span>
                <span>{msg['content']}</span>
                <span style="font-size:0.7em;color:#ccc; margin-left:5px;">[{time_str}]</span>
            </div>
        </div>
        """

def new_chat():
    ms.messages = []
    st.rerun()

def apply_theme(theme_name):
    tdict = ms.themes[theme_name]
    for key, val in tdict.items():
        if key.startswith("theme"):
            st._config.set_option(key, val)
    ms.themes["current_theme"] = theme_name
    ms.themes["refreshed"] = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙ Settings")
    if st.button("🔄 New Chat", use_container_width=True):
        new_chat()
    st.markdown("---")
    theme_choice = st.selectbox(
        "Select Theme",
        ("Light", "Dark"),
        index=0 if ms.themes["current_theme"]=="light" else 1
    )
    if theme_choice.lower() != ms.themes["current_theme"]:
        apply_theme(theme_choice.lower())
        st.rerun()
    st.markdown("---")
    st.markdown(f"**Current Theme:** {ms.themes['current_theme'].capitalize()}")

# --- CSS for themes ---
if ms.themes["current_theme"] == "dark":
    st.markdown(f"""
    <style>
    .stApp {{ background-color:{DARK_BG_COLOR} !important; color:{DARK_TEXT_COLOR} !important;}}
    .stChatInput > div {{ background-color:{INPUT_BG_COLOR} !important; border-radius:0.5rem; }}
    .stChatInput input {{ color:{INPUT_TEXT_COLOR} !important; background-color:{INPUT_BG_COLOR} !important; border: none !important; }}
    div[data-testid="stToolbar"], footer, .css-1n76c6l, div.css-1q8t5l3 {{
        background-color: {DARK_BG_COLOR} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color:#FFFFFF !important; color:#000000 !important;}
    .stChatInput > div { background-color:#F1F0F0 !important; border-radius:0.5rem;}
    .stChatInput input { color:black !important; background-color:transparent !important;}
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown(f"<h1 style='text-align:center;color:{PRIMARY_COLOR};'>🤖 Chatbot</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#555;'>Secure LLM</p>", unsafe_allow_html=True)

# --- Chat input ---
prompt = st.chat_input("Type your message here...")
if prompt:
    ms.messages.append({"role":"user", "content": prompt, "time": datetime.now()})
    try:
        response = process_prompt(prompt)
    except Exception as e:
        response = f"⚠ Error processing prompt:\n\n{e}"
    ms.messages.append({"role":"assistant", "content": response, "time": datetime.now()})

# --- Display messages + edit panel ---
for i, msg in enumerate(ms.messages):
    if msg["role"] == "user":
        st.markdown(format_message(msg, "user"), unsafe_allow_html=True)
        cols = st.columns([1,8])
        with cols[0]:
            fix_key = f"fix_btn_{i}"
            if st.button("✏", key=fix_key, help="Fix spelling / edit message"):
                ms[f"edit_open_{i}"] = True
        with cols[1]:
            st.write("")
        if ms.get(f"edit_open_{i}", False):
            suggested = simple_spell_suggest(msg.get("content",""))
            input_key = f"edit_input_{i}"
            corrected = st.text_input("Edit & resend:", value=suggested, key=input_key)
            resend_key = f"resend_{i}"
            if st.button("Resend corrected", key=resend_key):
                ms.messages.append({"role":"user", "content": corrected, "time": datetime.now()})
                try:
                    resp = process_prompt(corrected)
                except Exception as e:
                    resp = f"⚠ Error processing prompt:\n\n{e}"
                ms.messages.append({"role":"assistant", "content": resp, "time": datetime.now()})
                ms[f"edit_open_{i}"] = False
                st.rerun()
    else:
        st.markdown(format_message(msg, "assistant"), unsafe_allow_html=True)
