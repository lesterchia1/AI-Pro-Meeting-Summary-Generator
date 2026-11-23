"""
Pro AI Meeting Minutes Assistant — Streamlit Edition (Large File Support)
"""

import streamlit as st
from faster_whisper import WhisperModel
from groq import Groq
import os
import tempfile
import requests
from datetime import datetime
import re
import torch
import subprocess
import logging

# Setup professional logging format
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), "logs")
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

# --- API KEY CONFIGURATION ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    SEALION_API_KEY = st.secrets.get("SEALION_API_KEY", os.environ.get("SEALION_API_KEY"))
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    SEALION_API_KEY = os.environ.get("SEALION_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- OPTIONAL LIBRARIES SETUP ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# --- REGISTRIES ---
MODEL_MAP = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Qwen3.8-27b": "qwen/qwen3.8-27b",
    "Sea-Lion v4 32B (Qwen)": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
    "Sea-Lion v4 27B (Gemma)": "aisingapore/Gemma-SEA-LION-v4-27B-IT"
}

LANGUAGE_MAP = {
    "English": "en",
    "Malay": "ms",
    "Chinese": "zh",
    "Auto-Detect": None
}

# --- INITIALIZE STT ---
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("base", device=device, compute_type=compute_type)

stt_model = load_whisper_model()

TRANSCRIPT_DIR = os.path.join(os.getcwd(), "transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

def clean_text(text):
    return re.sub(r'[\*\*|_|=|#|`]', '', text).strip()

def extract_text_from_template(template_file):
    if template_file is None:
        return ""
    ext = os.path.splitext(template_file.name)[1].lower()
    text_content = ""
    try:
        if ext == ".docx" and DOCX_AVAILABLE:
            doc = Document(template_file)
            text_content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext == ".pdf" and PYPDF_AVAILABLE:
            reader = pypdf.PdfReader(template_file)
            text_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == ".txt":
            text_content = template_file.getvalue().decode("utf-8")
    except Exception as e:
        logging.error(f"Error parsing template file: {e}")
    return text_content

def call_llm(model_display_name, prompt):
    model_id = MODEL_MAP.get(model_display_name, "qwen/qwen3.8-27b")
    try:
        if "Sea-Lion" in model_display_name:
            url = "https://api.sea-lion.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {SEALION_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            return clean_text(response.json()['choices'][0]['message']['content'])
        else:
            if not groq_client:
                return "Error: Groq client not initialized (missing API key)."
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id
            )
            return clean_text(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"API processing exception: {str(e)}", exc_info=True)
        return f"LLM Error: {str(e)}"

def chunk_transcript(text, chunk_size=8000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def map_reduce_summary(full_transcript, model_choice, lang_name, custom_system_prompt, template_text):
    chunks = chunk_transcript(full_transcript)
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        chunk_prompt = f"Summarize this {lang_name} transcript segment:\n\n{chunk}"
        mini_summary = call_llm(model_choice, chunk_prompt)
        partial_summaries.append(mini_summary)

    base_instructions = custom_system_prompt.strip() if custom_system_prompt and custom_system_prompt.strip() else \
        "Synthesize the meeting transcript into an Executive Summary with Key Decisions and Action Items."
    
    template_instruction = f"\nFollow this template:\n{template_text.strip()}\n" if template_text and template_text.strip() else ""
    master_prompt = f"{base_instructions}\n{template_instruction}\nTranscript:\n" + "\n".join(partial_summaries)
    return call_llm(model_choice, master_prompt)

def save_transcript(labeled_segments, timestamp):
    filename = f"transcript_{timestamp}.txt"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Meeting Transcript — {timestamp}\n" + "=" * 60 + "\n\n")
        for (start, end, speaker, text) in labeled_segments:
            f.write(f"[{start:.1f}s – {end:.1f}s] {speaker}: {text.strip()}\n")
    return filepath

def generate_report(summary, transcript, format_choice, timestamp):
    if format_choice == "PDF" and REPORTLAB_AVAILABLE:
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.pdf")
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Meeting Summary", styles['Title']), Paragraph(summary.replace('\n', '<br/>'), styles['Normal'])]
        doc.build(elements)
        return path
    elif format_choice == "DOCX" and DOCX_AVAILABLE:
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.docx")
        doc = Document()
        doc.add_heading('Meeting Summary', 0)
        doc.add_paragraph(summary)
        doc.save(path)
        return path
    else:
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"SUMMARY:\n{summary}\n\n" + "=" * 30 + f"\n\nTRANSCRIPT:\n{transcript}")
        return path

# --- STREAMLIT UI ---
st.set_page_config(page_title="Pro AI Meeting Minutes", page_icon="🎙️", layout="wide")

st.title("🎙️ Pro AI Meeting Minutes Assistant — Streamlit Edition")

with st.sidebar:
    st.header("Configuration & Options")
    
    upload_input = st.file_uploader("Upload Video / Audio", type=["mp3", "wav", "m4a", "mp4", "mov"])
    record_input = st.audio_input("Or Record Audio")
    
    lang_name = st.selectbox("Speaking Language", list(LANGUAGE_MAP.keys()), index=0)
    model_choice = st.selectbox("AI Engine", list(MODEL_MAP.keys()), index=1)
    format_choice = st.radio("Export Format", ["TXT", "PDF", "DOCX"], index=0)
    noise_red = st.checkbox("Enable Noise Reduction (VAD)", value=True)
    
    st.markdown("---")
    st.subheader("Advanced Settings & Templates")
    custom_system_prompt = st.text_area("Custom System Prompt / Persona", placeholder="e.g. You are a strict project manager...")
    template_file_input = st.file_uploader("Upload Meeting Minute Template (.docx/.pdf/.txt)", type=[".docx", ".pdf", ".txt"])
    custom_vocab = st.text_input("Custom Vocabulary / Jargon Hint", placeholder="e.g. Project Orion, KPI dashboard")

    run_btn = st.button("🚀 Process Meeting", type="primary", use_container_width=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["AI Summary", "Raw Transcript", "Model Card & Info"])

audio_source = upload_input if upload_input is not None else record_input

if run_btn:
    if audio_source is None:
        st.error("Please upload an audio/video file or record audio first.")
    else:
        # Stream and save large files in chunks to protect system memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            for chunk in iter(lambda: audio_source.read(1024 * 1024), b""):
                tmp.write(chunk)
            temp_audio_path = tmp.name

        try:
            with st.spinner("🚀 Transcribing with Whisper base..."):
                lang_code = LANGUAGE_MAP.get(lang_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                segments_gen, _ = stt_model.transcribe(
                    temp_audio_path,
                    beam_size=1,
                    vad_filter=noise_red,
                    language=lang_code,
                    initial_prompt=custom_vocab.strip() if custom_vocab else None,
                    word_timestamps=False
                )
                seg_list = list(segments_gen)

            if not seg_list:
                st.warning("Transcript empty. Detected silence.")
            else:
                full_transcript = "\n".join([f"[{s.start:.1f}s] {s.text.strip()}" for s in seg_list])
                
                with st.spinner("🗺️ Extracting template and generating AI summary..."):
                    template_text = extract_text_from_template(template_file_input) if template_file_input else ""
                    final_summary = map_reduce_summary(full_transcript, model_choice, lang_name, custom_system_prompt, template_text)

                save_transcript([(s.start, s.end, "Speaker 1", s.text) for s in seg_list], timestamp)
                report_path = generate_report(final_summary, full_transcript, format_choice, timestamp)

                with tab1:
                    st.markdown("### Executive Summary")
                    st.markdown(final_summary)
                    
                    if report_path and os.path.exists(report_path):
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download Report ({format_choice})",
                                data=f,
                                file_name=os.path.basename(report_path),
                                mime="application/octet-stream"
                            )

                with tab2:
                    st.text_area("Transcription Output", full_transcript, height=400)

                st.success("✅ Meeting processing complete!")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logging.error(f"Error in meeting processing: {e}", exc_info=True)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
else:
    with tab1:
        st.info("Upload an audio file or use the recorder in the sidebar, then click **Process Meeting** to generate your summary.")
    with tab2:
        st.info("Raw transcript will appear here after processing.")
    with tab3:
        st.markdown("### Speed Optimizations\n* **STT:** Whisper `base` with INT8 computation\n* **Backend:** Streamlit native components with direct state management")
