"""
Pro AI Meeting Minutes Assistant — v5.0 (Streamlit Edition + SDLC Workflow + Template Ingestion + Multi-Model Hub)
"""

import streamlit as st
import os
import tempfile
import requests
from datetime import datetime
import re
import torch
import sqlite3
import pandas as pd
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup Page Configuration
st.set_page_config(
    page_title="Pro AI Meeting Minutes & SDLC Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup professional logging format
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"app_{datetime.now().strftime('%Y%m_%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- LOCAL DATABASE SETUP (Job Hunt Hub & Tracker) ---
DB_PATH = os.path.join(os.getcwd(), "tracker_hub.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_date TEXT,
            task_description TEXT,
            sdlc_phase TEXT,
            status TEXT,
            owner TEXT,
            target_date TEXT,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- MODEL REGISTRIES ---
MODEL_MAP = {
    "Groq: GPT-OSS 120B": "llama-3.3-70b-versatile",
    "Groq: GPT-OSS 20B": "llama-3.1-8b-instant",
    "AI Spore: Sea-Lion v4 27B": "sealion-v4-27b",
    "AI Spore: Sea-Lion v4 32B (Qwen)": "sealion-v4-32b-qwen",
    "HF: Qwen 3.5 2B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Local SLM (Ollama)": "local-ollama"
}

LANGUAGE_MAP = {
    "English": "en",
    "Malay": "ms",
    "Chinese": "zh",
    "Auto-Detect": None
}

SDLC_PHASES = ["Planning", "Requirements", "Design", "Implementation", "Testing", "Deployment", "Maintenance"]
ACTION_STATES = ["Follow-up Required", "Pending Reply", "Accepted", "Rejected", "In Progress", "Completed"]

# --- INITIALIZE STT (Optimized for Speed & Noise) ---
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    from faster_whisper import WhisperModel
    return WhisperModel("medium", device=device, compute_type=compute_type)

try:
    stt_model = load_whisper_model()
except Exception as e:
    st.warning(f"Could not load GPU-accelerated Whisper model locally. Falling back or running on CPU: {e}")
    stt_model = None

def clean_text(text):
    return re.sub(r'[\*\*|_|=|#|`]', '', text).strip()

def call_llm(model_display_name, prompt):
    try:
        if model_display_name == "Local SLM (Ollama)":
            url = "http://localhost:11434/api/generate"
            payload = {"model": "llama3", "prompt": prompt, "stream": False}
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return clean_text(response.json().get('response', ''))
            else:
                return f"Local Ollama Error: {response.text}"
                
        elif "Groq" in model_display_name:
            from groq import Groq
            # Pulling Groq API key securely from Streamlit secrets or environment variables
            groq_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.environ.get("GROQ_API_KEY")
            client = Groq(api_key=groq_key)
            model_id = MODEL_MAP.get(model_display_name, "llama-3.3-70b-versatile")
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id
            )
            return clean_text(chat_completion.choices[0].message.content)
            
        elif "AI Spore" in model_display_name or "HF" in model_display_name:
            hf_key = st.secrets.get("HF_API_KEY") if "HF_API_KEY" in st.secrets else os.environ.get("HF_API_KEY")
            headers = {"Authorization": f"Bearer {hf_key}"} if hf_key else {}
            api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-1.5B-Instruct"
            payload = {"inputs": prompt}
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                res_json = response.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    return clean_text(res_json[0].get('generated_text', str(res_json)))
                return str(res_json)
            else:
                return f"API Inference Error ({response.status_code}): {response.text}"
                
    except Exception as e:
        logging.error(f"LLM inference exception: {str(e)}", exc_info=True)
        return f"LLM Error: {str(e)}"

def extract_template_text(uploaded_file):
    if uploaded_file is None:
        return None
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        if ext == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(tmp_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif ext in [".docx", ".doc"]:
            from docx import Document
            doc = Document(tmp_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        os.unlink(tmp_path)
    except Exception as e:
        logging.error(f"Error parsing template: {e}")
    return text.strip()

def send_email_report(sender_email, sender_password, receiver_email, smtp_server, smtp_port, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, int(smtp_port))
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration & Settings")

model_choice = st.sidebar.selectbox("Select AI Model", list(MODEL_MAP.keys()), index=0)
lang_name = st.sidebar.selectbox("Speaking Language", list(LANGUAGE_MAP.keys()), index=0)
noise_red = st.sidebar.checkbox("Advanced Spectral Noise Reduction & VAD", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Custom Meeting Template")
template_file = st.sidebar.file_uploader("Upload Company/Project Template (PDF/DOCX)", type=["pdf", "docx"])

st.sidebar.markdown("---")
st.sidebar.subheader("📧 Email Dispatch Settings")
enable_email = st.sidebar.checkbox("Enable Email Dispatch", value=False)
smtp_server = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
smtp_port = st.sidebar.text_input("SMTP Port", value="587")
sender_email = st.sidebar.text_input("Sender Email", value="")
sender_password = st.sidebar.text_input("Sender Password / App Key", type="password", value="")
receiver_email = st.sidebar.text_input("Recipient Email", value="")

# --- MAIN APP LAYOUT ---
st.title("🎙️ Pro AI Meeting Minutes Assistant — v5.0")
st.markdown("Automated Transcription, Custom Layout Structuring, SDLC Action Items, and Project Tracker Hub.")

tab1, tab2, tab3 = st.tabs(["🚀 Audio Processing & Minutes", "📊 Job Hunt Hub & SDLC Tracker", "✉️ Email Dispatch"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Audio Source")
        audio_file = st.file_uploader("Upload Audio / Video File", type=["wav", "mp3", "m4a", "mp4", "webm"])
        
        st.info("Tip: You can upload voice notes recorded directly from your phone, Zoom, or Teams.")
        
        process_btn = st.button("🚀 Process Meeting & Log to Tracker", type="primary")

    with col2:
        st.subheader("2. Real-Time Status & Summary")
        summary_placeholder = st.empty()

    if process_btn:
        if audio_file is None:
            st.error("Please upload an audio or video file first.")
        elif stt_model is None:
            st.error("Faster-whisper model failed to initialize.")
        else:
            with st.spinner("Processing audio, applying noise reduction & transcribing..."):
                try:
                    suffix = os.path.splitext(audio_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(audio_file.getvalue())
                        audio_path = tmp.name

                    lang_code = LANGUAGE_MAP.get(lang_name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if noise_red:
                        try:
                            import librosa
                            import soundfile as sf
                            import noisereduce as nr
                            data, rate = librosa.load(audio_path, sr=None)
                            reduced_noise = nr.reduce_noise(y=data, sr=rate)
                            sf.write(audio_path, reduced_noise, rate)
                        except Exception as n_err:
                            logging.warning(f"Noise reduction skipped: {n_err}")

                    segments_gen, _ = stt_model.transcribe(
                        audio_path,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        language=lang_code,
                        task="transcribe"
                    )
                    seg_list = list(segments_gen)
                    os.unlink(audio_path)

                    if not seg_list:
                        st.warning("Transcript empty. Detected silence or low audio quality.")
                    else:
                        full_transcript = "\n".join([f"[{s.start:.1f}s] {s.text.strip()}" for s in seg_list])
                        
                        template_instructions = ""
                        if template_file is not None:
                            tpl_text = extract_template_text(template_file)
                            if tpl_text:
                                template_instructions = f"\n\nFormat your output strictly to follow this template structure:\n{tpl_text}\n"

                        master_prompt = f"""
You are an expert AI Chief of Staff and Technical Project Manager aligned with SDLC workflows and project tracking standards.
Analyze the meeting transcript below and generate comprehensive minutes in {lang_name}.
{template_instructions}
Ensure your output includes:
1. Executive Overview
2. Key Decisions & Architecture Updates
3. Action Items formatted clearly with Assigned Owner, Target Date, SDLC Phase ({', '.join(SDLC_PHASES)}), and Status ({', '.join(ACTION_STATES)}).
4. Open Risks / Blockers

Transcript:
{full_transcript}
"""
                        final_summary = call_llm(model_choice, master_prompt)

                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO action_items (meeting_date, task_description, sdlc_phase, status, owner, target_date, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (timestamp, "Review meeting notes and action items", "Implementation", "In Progress", "Project Lead", "Next Sprint", "Auto-logged from session")
                        )
                        conn.commit()
                        conn.close()

                        st.success("Meeting processed and logged to Job Hunt / Project Tracker Hub successfully!")
                        
                        st.markdown("### 📑 Generated Meeting Minutes")
                        st.markdown(final_summary)
                        
                        st.markdown("### 📝 Raw Transcript")
                        st.text_area("Transcript Output", full_transcript, height=200)

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    logging.error(f"Streamlit processing error: {e}", exc_info=True)

with tab2:
    st.subheader("📊 Job Hunt Hub & Project Tracker Dashboard")
    
    conn = sqlite3.connect(DB_PATH)
    df_tracker = pd.read_sql_query("SELECT * FROM action_items ORDER BY id DESC", conn)
    conn.close()

    if not df_tracker.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_phase = st.selectbox("Filter by SDLC Phase", ["All"] + SDLC_PHASES)
        with col_f2:
            selected_status = st.selectbox("Filter by Lifecycle State", ["All"] + ACTION_STATES)

        filtered_df = df_tracker.copy()
        if selected_phase != "All":
            filtered_df = filtered_df[filtered_df['sdlc_phase'] == selected_phase]
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df['status'] == selected_status]

        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Tracker as CSV",
            data=csv_data,
            file_name=f"tracker_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    else:
        st.info("No records found in the tracker database yet. Process a meeting to populate items.")

with tab3:
    st.subheader("✉️ Email Meeting Minutes & Action Items")
    email_subject = st.text_input("Email Subject", value=f"Meeting Minutes & Action Items - {datetime.now().strftime('%Y-%m-%d')}")
    email_body = st.text_area("Email Content / Summary", height=300, value="Please find attached/above the meeting summary and action items.")
    
    if st.button("📧 Send Email"):
        if not enable_email:
            st.warning("Please enable email dispatch in the sidebar settings.")
        elif not receiver_email or not sender_email or not sender_password:
            st.error("Please fill in all email credentials in the sidebar.")
        else:
            success, message = send_email_report(
                sender_email, sender_password, receiver_email, smtp_server, smtp_port, email_subject, email_body
            )
            if success:
                st.success(message)
            else:
                st.error(message)
