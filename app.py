"""
Pro AI Meeting Minutes Assistant — v4.1 (Streamlit Edition)
Conversion of the original Gradio app to Streamlit.
All features preserved.
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
import gc
from tenacity import retry, stop_after_attempt, wait_exponential
import subprocess
import logging
import time

# ----------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# API KEYS (secrets + env fallback)
# ----------------------------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SEALION_API_KEY = st.secrets["SEALION_API_KEY"]
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    SEALION_API_KEY = os.environ.get("SEALION_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not GROQ_API_KEY or not SEALION_API_KEY:
    st.error("Missing required API keys. Please set GROQ_API_KEY and SEALION_API_KEY in secrets or environment.")
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------------------------------------------
# OPTIONAL LIBRARIES
# ----------------------------------------------------------------------
REPORTLAB_AVAILABLE = False
DOCX_AVAILABLE = False
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    pass

try:
    from docx import Document
    from docx.shared import Pt
    DOCX_AVAILABLE = True
except ImportError:
    pass

# ----------------------------------------------------------------------
# SPEAKER DIARIZATION (pyannote)
# ----------------------------------------------------------------------
DIARIZATION_AVAILABLE = False
diarize_pipeline = None
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    if HF_TOKEN:
        diarize_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN
        )
        if torch.cuda.is_available():
            diarize_pipeline = diarize_pipeline.to(torch.device("cuda"))
        DIARIZATION_AVAILABLE = True
        logging.info("Diarization pipeline loaded successfully.")
    else:
        logging.warning("HF_TOKEN not set – diarization disabled.")
except ImportError:
    logging.info("pyannote.audio not installed – diarization disabled.")
except Exception as e:
    logging.error(f"Diarization initialization error: {e}")

# ----------------------------------------------------------------------
# REGISTRIES
# ----------------------------------------------------------------------
MODEL_MAP = {
    "Llama-3.3 70B": "llama-3.3-70b-versatile",
    "Llama-3.1 8B": "llama-3.1-8b-instant",
    "Sea-Lion v4 27B": "aisingapore/Gemma-SEA-LION-v4-27B-IT"
}

LANGUAGE_MAP = {
    "English": "en",
    "Malay": "ms",
    "Chinese": "zh",
    "Auto-Detect": None
}

# ----------------------------------------------------------------------
# INIT STT
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
stt_model = WhisperModel("small", device=device, compute_type=compute_type)

# ----------------------------------------------------------------------
# TRANSCRIPT SAVE FOLDER
# ----------------------------------------------------------------------
TRANSCRIPT_DIR = os.path.join(os.getcwd(), "transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def clean_text(text):
    return re.sub(r'[\*\*|_|=|#|`]', '', text).strip()

def call_llm(model_display_name, prompt):
    model_id = MODEL_MAP.get(model_display_name)
    logging.info(f"Calling LLM: {model_display_name} ({model_id})")
    try:
        if "Sea-Lion" in model_display_name:
            url = "https://api.sea-lion.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {SEALION_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            return clean_text(response.json()['choices'][0]['message']['content'])
        else:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id
            )
            return clean_text(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"LLM error: {e}", exc_info=True)
        return f"LLM Error: {str(e)}"

def convert_audio_to_standard_format(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "-y", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None

def run_diarization(audio_path):
    if not DIARIZATION_AVAILABLE or diarize_pipeline is None:
        return []
    try:
        converted = convert_audio_to_standard_format(audio_path)
        audio_to_use = converted if converted else audio_path
        output = diarize_pipeline(audio_to_use)
        segments = []
        if hasattr(output, 'speaker_diarization'):
            for turn, speaker in output.speaker_diarization:
                segments.append((turn.start, turn.end, speaker))
        elif hasattr(output, 'exclusive_speaker_diarization'):
            for turn, speaker in output.exclusive_speaker_diarization:
                segments.append((turn.start, turn.end, speaker))
        if converted and os.path.exists(converted):
            os.remove(converted)
        return segments
    except Exception as e:
        logging.error(f"Diarization error: {e}", exc_info=True)
        return []

def assign_speakers(whisper_segments, diarization_segments):
    if not diarization_segments:
        return [(s.start, s.end, "Speaker ?", s.text) for s in whisper_segments]
    result = []
    for s in whisper_segments:
        best_speaker = "Speaker ?"
        best_overlap = 0.0
        for (d_start, d_end, label) in diarization_segments:
            overlap = max(0, min(s.end, d_end) - max(s.start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = label
        result.append((s.start, s.end, best_speaker, s.text))
    return result

def chunk_transcript(text, chunk_size=8000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def map_reduce_summary(full_transcript, model_choice, lang_name, status_placeholder):
    chunks = chunk_transcript(full_transcript)
    partial_summaries = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        status_placeholder.write(f"📦 Summarising part {i+1}/{total_chunks}...")
        chunk_prompt = (
            f"You are summarising a segment of a {lang_name} meeting transcript. "
            f"Be concise. Extract key facts, decisions, and action items:\n\n{chunk}"
        )
        mini_summary = call_llm(model_choice, chunk_prompt)
        partial_summaries.append(f"[Segment {i+1}]\n{mini_summary}")
    status_placeholder.write("📑 Synthesising master summary (Reduce)...")
    combined_notes = "\n\n".join(partial_summaries)
    master_prompt = f"""
You are a professional Chief of Staff. Below are summaries of different segments of a long meeting.
Synthesise them into one cohesive Master Executive Summary in {lang_name}.

Structure your output with these four sections:
1. Executive Overview — what was the meeting about?
2. Key Decisions — what was agreed upon?
3. Action Items — who is doing what, and by when?
4. Open Questions — anything left unresolved?

Segment Summaries:
{combined_notes}
"""
    return call_llm(model_choice, master_prompt)

def save_transcript(labeled_segments, timestamp):
    filename = f"transcript_{timestamp}.txt"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Meeting Transcript — {timestamp}\n")
        f.write("="*60 + "\n\n")
        for (start, end, speaker, text) in labeled_segments:
            f.write(f"[{start:.1f}s – {end:.1f}s]  {speaker}:  {text.strip()}\n")
    return filepath

def generate_report(summary, transcript, format_choice, timestamp):
    if format_choice == "PDF":
        if not REPORTLAB_AVAILABLE:
            return None
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.pdf")
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        body_style = ParagraphStyle('Body', parent=styles['Normal'], spaceAfter=10)
        elements = [
            Paragraph("Meeting Summary", styles['Title']),
            Paragraph(summary.replace('\n', '<br/>'), body_style),
            Spacer(1, 12),
            Paragraph("Full Transcript", styles['Heading2']),
            Paragraph(transcript.replace('\n', '<br/>'), styles['Code'])
        ]
        doc.build(elements)
        return path
    elif format_choice == "DOCX":
        if not DOCX_AVAILABLE:
            return None
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.docx")
        doc = Document()
        doc.add_heading('Meeting Summary', 0)
        doc.add_paragraph(summary)
        doc.add_page_break()
        doc.add_heading('Full Transcript', level=1)
        doc.add_paragraph(transcript)
        doc.save(path)
        return path
    else:  # TXT
        path = os.path.join(tempfile.gettempdir(), f"report_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"SUMMARY:\n{summary}\n\n" + "="*30 + f"\n\nTRANSCRIPT:\n{transcript}")
        return path

# ----------------------------------------------------------------------
# MAIN PROCESSING FUNCTION
# ----------------------------------------------------------------------
def process_meeting(
    audio_file,          # UploadedFile or bytes from recorder
    is_recording,        # bool: True if from audio_input
    lang_name,
    model_choice,
    noise_red,
    format_choice,
    custom_vocab,
    enable_diarization,
    status_placeholder
):
    """
    Core processing.
    Returns (full_transcript, summary, report_bytes, report_filename, error_msg)
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav" if is_recording else ".tmp") as tmp:
            if is_recording:
                # audio_file is bytes (from st.audio_input)
                tmp.write(audio_file)
            else:
                # audio_file is UploadedFile (from st.file_uploader)
                tmp.write(audio_file.getvalue())
            tmp_path = tmp.name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lang_code = LANGUAGE_MAP.get(lang_name)
        initial_prompt = custom_vocab.strip() if custom_vocab else None

        # ---- Transcription ----
        status_placeholder.write("🚀 Transcribing...")
        segments_gen, info = stt_model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=noise_red,
            language=lang_code,
            task="transcribe",
            initial_prompt=initial_prompt
        )
        seg_list = list(segments_gen)
        if not seg_list:
            return None, None, None, None, "Transcript empty. The AI detected silence. Please check your audio source."

        # ---- Diarization ----
        diarization_segs = []
        if enable_diarization and DIARIZATION_AVAILABLE:
            status_placeholder.write("🎙️ Identifying speakers (diarization)...")
            diarization_segs = run_diarization(tmp_path)

        labeled_segments = assign_speakers(seg_list, diarization_segs)

        # Build full transcript
        full_transcript = ""
        for (start, end, speaker, text) in labeled_segments:
            if diarization_segs:
                full_transcript += f"[{start:.1f}s] {speaker}: {text.strip()}\n"
            else:
                full_transcript += f"[{start:.1f}s] {text.strip()}\n"

        # ---- Summary ----
        status_placeholder.write("🗺️ Starting Map-Reduce summarisation...")
        final_summary = map_reduce_summary(full_transcript, model_choice, lang_name, status_placeholder)

        # ---- Save transcript ----
        status_placeholder.write("💾 Saving transcript...")
        save_transcript(labeled_segments, timestamp)

        # ---- Generate report ----
        status_placeholder.write("📄 Generating downloadable report...")
        report_path = generate_report(final_summary, full_transcript, format_choice, timestamp)
        if report_path:
            with open(report_path, "rb") as f:
                report_bytes = f.read()
            os.remove(report_path)
            report_filename = os.path.basename(report_path)
        else:
            report_bytes = None
            report_filename = None

        # Cleanup temp audio
        os.unlink(tmp_path)

        diarize_note = ""
        if enable_diarization and not DIARIZATION_AVAILABLE:
            diarize_note = "\n\n⚠️ Speaker diarization was requested but pyannote.audio is not installed or HF_TOKEN is missing."
        elif enable_diarization and DIARIZATION_AVAILABLE and not diarization_segs:
            diarize_note = "\n\n⚠️ Diarization ran but returned no speaker segments."

        return full_transcript + diarize_note, final_summary, report_bytes, report_filename, None

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return None, None, None, None, f"Error: {str(e)}"

# ----------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Pro AI Meeting Minutes", layout="wide")

st.markdown(
    """
    # 🎙️ Pro AI Meeting Minutes — v4.1
    _Streamlit Edition – Model card · Log File · Visual System Map_
    """
)

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload Video / Audio",
        type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"],
        help="Upload any common audio or video file."
    )
    st.markdown("— or —")
    recorded_audio = st.audio_input("Record Live Audio")

    st.header("Options")
    lang_dropdown = st.selectbox(
        "Speaking Language",
        list(LANGUAGE_MAP.keys()),
        index=0
    )
    model_dropdown = st.selectbox(
        "AI Engine",
        list(MODEL_MAP.keys()),
        index=0
    )
    format_choice = st.radio(
        "Export Format",
        ["TXT", "PDF", "DOCX"],
        index=0,
        horizontal=True
    )
    noise_red = st.checkbox("Enable Noise Reduction (VAD)", value=True)

    with st.expander("Advanced"):
        custom_vocab = st.text_area(
            "Custom Vocabulary / Jargon Hint",
            placeholder="e.g. Project Orion, KPI dashboard, SCRUM",
            help="Words Whisper might mishear. Separate with commas."
        )
        diarization_available_str = "✅ (pyannote loaded)" if DIARIZATION_AVAILABLE else "⚠️ (not available — set HF_TOKEN)"
        enable_diarization = st.checkbox(
            f"Enable Speaker Diarization {diarization_available_str}",
            value=False,
            help="Labels each line with the speaker. Requires GPU for best performance."
        )

    process_button = st.button("🚀 Process Meeting", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# PROCESSING LOGIC
# ----------------------------------------------------------------------
if process_button:
    if uploaded_file is None and recorded_audio is None:
        st.warning("Please upload a file or record audio first.")
    else:
        # Determine which source to use: if both are provided, prefer uploaded file
        if uploaded_file is not None:
            audio_source = uploaded_file
            is_recording = False
        else:
            audio_source = recorded_audio  # This is bytes
            is_recording = True

        with st.status("Processing meeting...", expanded=True) as status:
            transcript, summary, report_bytes, report_filename, error = process_meeting(
                audio_source,
                is_recording,
                lang_dropdown,
                model_dropdown,
                noise_red,
                format_choice,
                custom_vocab,
                enable_diarization,
                status
            )

        if error:
            st.error(error)
        else:
            # Display outputs in tabs
            tab1, tab2 = st.tabs(["📝 AI Summary", "📄 Raw Transcript"])
            with tab1:
                st.markdown(summary)
            with tab2:
                st.text_area("Transcript", transcript, height=400)

            # Download button
            if report_bytes and report_filename:
                st.download_button(
                    label=f"📥 Download Report ({format_choice})",
                    data=report_bytes,
                    file_name=report_filename,
                    mime="application/octet-stream"
                )

# ----------------------------------------------------------------------
# MODEL CARD (always visible at bottom)
# ----------------------------------------------------------------------
with st.expander("ℹ️ Model Card & Info"):
    st.markdown("""
    ### Model Card & Info

    #### 1. Speech-to-Text Engine (ASR: FastWhisper)
    * **Model Details:** Whisper small-v3 optimised via CTranslate2 (MIT/Apache 2.0). **WER:** 1.59% (internal tests: 3.37%, 2.02%, 0.28%, 0.69%).
    * **Intended Use:** High-speed, low-latency multilingual transcription.
    * **Out-of-Scope:** Not for real-time streaming without chunking, nor for speaker diarisation/emotion detection.
    * **Intent:** Deliver 4x faster inference on CPU/GPU while preserving accuracy.
    * **Limitations:** Higher WER on heavy accents, extreme noise, or audio <0.5 seconds.

    #### 2. Llama-3.3 70B
    * **Model Details:** Dec 6, 2024 | Auto-regressive (GQA) | Llama 3.3 Community License | 128k ctx | ~15T tokens (cutoff Dec 2023).
    * **Intended Use:** Complex reasoning, coding, multilingual chat, synthetic data generation.
    * **Out-of-Scope:** Strictly for 8 supported languages (Eng, Ger, Fre, Ita, Por, Hin, Spa, Thai) without fine-tuning. Not for violation of laws/AUP.
    * **Intent:** SOTA 70B model beating many closed-source models (MMLU 86.0%, MATH 77.0%).
    * **Limitations:** Unpredictable outputs, cutoff Dec 2023, requires safety guardrails (Llama Guard 3).

    #### 3. Llama-3.1 8B
    * **Model Details:** July 23, 2024 | Auto-regressive (GQA) | Llama 3.1 Community License | 128k ctx | ~15T tokens.
    * **Intended Use:** Lightweight apps, RAG, on-device assistants, summarization.
    * **Out-of-Scope:** Not for complex reasoning (>70B tasks). Restricted to 8 supported languages without fine-tuning.
    * **Intent:** Balanced performance/resource for consumer-grade hardware (MMLU 69.4%, HumanEval 72.6%).
    * **Limitations:** Lower factual accuracy vs 70B; performance drops on non-supported languages.

    #### 4. Sea-Lion v4 27B (Gemma-SEA-LION)
    * **Model Details:** Aug 25, 2025 | Decoder-only (Gemma 3 based) | Gemma Terms of Use | 128k ctx | Multimodal | Trained on ~500B tokens across 11 SEA languages (Burmese, Thai, Viet, Malay, etc.).
    * **Intended Use:** Text generation & visual Q&A optimised for Southeast Asia.
    * **Out-of-Scope:** **WARNING:** Not safety-aligned. Not for adversarial prompting or critical infra without fine-tuning.
    * **Intent:** Outperforms <200B open models on SEA benchmarks, rivaling top closed models.
    * **Limitations:** Not safety aligned (needs developer safeguards), hallucinations, vision limited to Gemma 3 capabilities.
    """)
