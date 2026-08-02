"""
Pro AI Meeting Minutes Assistant — v4.1
Improvements over v1:
  • Map-Reduce summarisation — handles 2-hour meetings without data loss
  • Speaker Diarization — pyannote.audio (optional, GPU-friendly, graceful fallback)
  • Transcript auto-saved to /transcripts folder with speaker labels
  • Custom vocabulary hint (Whisper initial_prompt)
  • All v1 features preserved (languages, models, export formats, noise reduction)
  • Added in Visual System Map, Model Card and Log File.
"""

import gradio as gr
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
import subprocess  # ✅ ADD THIS IMPORT -New
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
        logging.StreamHandler() # Also prints to your terminal window
    ]
)


# ---  API KEY CONFIGURATION (Cross-Platform: Colab + Hugging Face) ---

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    SEALION_API_KEY = userdata.get('SEALION_API_KEY')
    HF_TOKEN = userdata.get('HF_TOKEN')
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    SEALION_API_KEY = os.environ.get("SEALION_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not GROQ_API_KEY or not SEALION_API_KEY:
    raise ValueError("Missing required API keys. Please check your configuration.")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY)


# --- OPTIONAL LIBRARIES SETUP ---
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Pt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# --- OPTIONAL: Speaker Diarization ---
DIARIZATION_AVAILABLE = False
diarize_pipeline = None
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        diarize_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN  
        )
        if torch.cuda.is_available():
            diarize_pipeline = diarize_pipeline.to(torch.device("cuda"))
        
        DIARIZATION_AVAILABLE = True
        print("✅ Diarization pipeline loaded (community model)")
    else:
        print("⚠️ HF_TOKEN not set — speaker diarization disabled.")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        print("   Then accept terms at: https://huggingface.co/pyannote/speaker-diarization-community-1")
        
except ImportError:
    print("ℹ️ pyannote.audio not installed — diarization disabled.")
    print("   Install with: pip install pyannote.audio==4.0.4")
except Exception as e:
    print(f"⚠️ Diarization initialization error: {e}")
    DIARIZATION_AVAILABLE = False


# --- 2. REGISTRIES ---
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

# --- 3. INITIALIZE STT ---
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
stt_model = WhisperModel("small", device=device, compute_type=compute_type)

# --- 4. TRANSCRIPT SAVE FOLDER ---
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.getcwd()

TRANSCRIPT_DIR = os.path.join(base_path, "transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPER: Clean LLM output
# ─────────────────────────────────────────────
def clean_text(text):
    return re.sub(r'[\*\*|_|=|#|`]', '', text).strip()


# ─────────────────────────────────────────────
# HELPER: LLM call (Unified & Fixed Version)
# ─────────────────────────────────────────────
def call_llm(model_display_name, prompt):
    model_id = MODEL_MAP.get(model_display_name)
    logging.info(f"Dispatching inference call to model: {model_display_name} ({model_id})")
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
            logging.info("Sea-Lion LLM inference successfully completed.")
            return clean_text(response.json()['choices'][0]['message']['content'])
        else:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id
            )
            logging.info("Groq LLM inference successfully completed.")
            return clean_text(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"API processing exception on model {model_display_name}: {str(e)}", exc_info=True)
        return f"LLM Error: {str(e)}"


# ─────────────────────────────────────────────
# AUDIO PRE-PROCESSING FORMAT CONVERSION
# ─────────────────────────────────────────────
def convert_audio_to_standard_format(input_path, output_path=None):
    """
    Convert audio to standard 16kHz mono WAV format.
    This resolves sample count mismatches.
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        "-y",            # overwrite output
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Audio normalization successful. Temp target: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg pipeline failure: {e.stderr.decode('utf-8')}")
        return None


# ─────────────────────────────────────────────
# Speaker Diarization
# ─────────────────────────────────────────────
def run_diarization(audio_path):
    """
    Returns a list of (start_sec, end_sec, speaker_label) tuples.
    Falls back to empty list if diarization is unavailable.
    """
    if not DIARIZATION_AVAILABLE or diarize_pipeline is None:
        return []
    try:
        print(f"Original audio: {audio_path}")
        converted_path = convert_audio_to_standard_format(audio_path)
        
        if converted_path is None:
            print("⚠️ Could not convert audio, trying original...")
            audio_to_use = audio_path
        else:
            audio_to_use = converted_path
            print(f"Using converted audio: {audio_to_use}")
        
        output = diarize_pipeline(audio_to_use)
        segments = []
        
        if hasattr(output, 'speaker_diarization'):
            for turn, speaker in output.speaker_diarization:
                segments.append((turn.start, turn.end, speaker))
            print(f"✅ Found {len(segments)} speaker segments")
        elif hasattr(output, 'exclusive_speaker_diarization'):
            for turn, speaker in output.exclusive_speaker_diarization:
                segments.append((turn.start, turn.end, speaker))
            print(f"✅ Found {len(segments)} exclusive speaker segments")
        else:
            print(f"⚠️ Unknown output type: {type(output)}")
        
        if converted_path and converted_path != audio_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                print(f"Cleaned up: {converted_path}")
            except:
                pass
        
        return segments
        
    except Exception as e:
        print(f"Diarization error: {e}")
        import traceback
        traceback.print_exc()
        return []


def assign_speakers(whisper_segments, diarization_segments):
    """Merges Whisper text segments with pyannote speaker labels."""
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


# ─────────────────────────────────────────────
# Map-Reduce chunking & Summarisation
# ─────────────────────────────────────────────
def chunk_transcript(text, chunk_size=8000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def map_reduce_summary(full_transcript, model_choice, lang_name, progress, base_progress=0.65):
    chunks = chunk_transcript(full_transcript)
    partial_summaries = []

    # MAP
    for i, chunk in enumerate(chunks):
        frac = base_progress + (i / max(len(chunks), 1)) * 0.20
        progress(frac, desc=f"📦 Summarising part {i + 1} of {len(chunks)}...")
        chunk_prompt = (
            f"You are summarising a segment of a {lang_name} meeting transcript. "
            f"Be concise. Extract key facts, decisions, and any action items:\n\n{chunk}"
        )
        mini_summary = call_llm(model_choice, chunk_prompt)
        partial_summaries.append(f"[Segment {i + 1}]\n{mini_summary}")

    # REDUCE
    progress(0.88, desc="📑 Synthesising master summary (Reduce)...")
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


# ─────────────────────────────────────────────
# Save transcript to disk
# ─────────────────────────────────────────────
def save_transcript(labeled_segments, timestamp):
    filename = f"transcript_{timestamp}.txt"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Meeting Transcript — {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        for (start, end, speaker, text) in labeled_segments:
            f.write(f"[{start:.1f}s – {end:.1f}s]  {speaker}:  {text.strip()}\n")
    print(f"✅ Transcript saved → {filepath}")
    return filepath


# ─────────────────────────────────────────────
# EXPORT LOGIC
# ─────────────────────────────────────────────
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
            f.write(f"SUMMARY:\n{summary}\n\n" + "=" * 30 + f"\n\nTRANSCRIPT:\n{transcript}")
        return path


# ─────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────
def process_meeting(
    file_path,
    record_path,
    lang_name,
    model_choice,
    noise_red,
    format_choice,
    custom_vocab,          
    enable_diarization,    
    progress=gr.Progress()
):
    source = file_path if file_path is not None else record_path
    if source is None:
        logging.warning("Process lifecycle initiated without valid audio input resource.")
        return "No audio/video detected.", "Please upload a file or record audio first.", None

    logging.info(f"Processing session started. Source file target: {source} | Language choice: {lang_name}")
    lang_code = LANGUAGE_MAP.get(lang_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_source = str(source)

    try:
        # ── STAGE 1: Transcription ────────────────
        progress(0.05, desc=f"🚀 Transcribing ({lang_name})...")
        initial_prompt = custom_vocab.strip() if custom_vocab else None

        segments_gen, info = stt_model.transcribe(
            audio_source,
            beam_size=5,
            vad_filter=noise_red,
            language=lang_code,
            task="transcribe",
            initial_prompt=initial_prompt
        )

        seg_list = list(segments_gen)
        if not seg_list:
            return "Transcript empty.", "The AI detected silence. Please check your audio source.", None

        for i, s in enumerate(seg_list):
            progress(0.05 + (i / max(len(seg_list), 1)) * 0.45, desc="Transcribing...")

        # ── STAGE 2 (Optional): Speaker Diarization ──
        diarization_segs = []
        if enable_diarization and DIARIZATION_AVAILABLE:
            progress(0.52, desc="🎙️ Identifying speakers (diarization)...")
            diarization_segs = run_diarization(audio_source)

        labeled_segments = assign_speakers(seg_list, diarization_segs)

        full_transcript = ""
        for (start, end, speaker, text) in labeled_segments:
            if diarization_segs:
                full_transcript += f"[{start:.1f}s] {speaker}: {text.strip()}\n"
            else:
                full_transcript += f"[{start:.1f}s] {text.strip()}\n"

        # ── STAGE 3 + 4: Map-Reduce Summary ──────
        progress(0.60, desc="🗺️ Starting Map-Reduce summarisation...")
        final_summary = map_reduce_summary(full_transcript, model_choice, lang_name, progress)

        # ── STAGE 5: Save transcript to disk ─────
        progress(0.92, desc="💾 Saving transcript to /transcripts...")
        save_transcript(labeled_segments, timestamp)

        # ── STAGE 6: Export report ────────────────
        progress(0.96, desc="📄 Generating downloadable report...")
        report_path = generate_report(final_summary, full_transcript, format_choice, timestamp)

        diarize_note = ""
        if enable_diarization and not DIARIZATION_AVAILABLE:
            diarize_note = "\n\n⚠️ Speaker diarization was requested but pyannote.audio is not installed or HF_TOKEN is missing. Install it and set HF_TOKEN to enable this feature.\n"
        elif enable_diarization and DIARIZATION_AVAILABLE and not diarization_segs:
            diarize_note = "\n\n⚠️ Diarization ran but returned no speaker segments. This can happen with very short audio.\n"

        progress(1.0, desc="✅ Done!")
        return full_transcript + diarize_note, final_summary, report_path

    except Exception as e:
        import traceback
        logging.error(f"Error in meeting processing lifecycle: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", "Process failed. Check terminal for logs.", None


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Pro AI Meeting Minutes — v4.1\n"
        "_Model card · Log File · Visual System Map_"
    )

    with gr.Row():
        with gr.Column(scale=1):
            upload_input = gr.File(
                label="Upload Video / Audio",
                file_types=["audio", "video"]
            )
            record_input = gr.Audio(
                label="OR Record Live Audio",
                type="filepath"
            )

            gr.Markdown("### Options")
            lang_dropdown = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()),
                value="English",
                label="Speaking Language"
            )
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="Llama-3.3 70B",
                label="AI Engine"
            )
            format_choice = gr.Radio(
                choices=["TXT", "PDF", "DOCX"],
                value="TXT",
                label="Export Format"
            )
            noise_red = gr.Checkbox(
                value=True,
                label="Enable Noise Reduction (VAD)"
            )

            gr.Markdown("### Advanced")
            custom_vocab = gr.Textbox(
                label="Custom Vocabulary / Jargon Hint",
                placeholder="e.g. Project Orion, KPI dashboard, SCRUM",
                lines=2,
                info="Words Whisper might mishear. Separate with commas."
            )
            enable_diarization = gr.Checkbox(
                value=False,
                label=f"Enable Speaker Diarization {'✅ (pyannote loaded)' if DIARIZATION_AVAILABLE else '⚠️ (pyannote not available — set HF_TOKEN)'}",
                info="Labels each line with the speaker. Requires GPU for best performance."
            )

            run_btn = gr.Button("🚀 Process Meeting", variant="primary")
            download_output = gr.File(label="Download Report")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("AI Summary"):
                    output_summary = gr.Markdown(value="*Summary will appear here...*")
                with gr.TabItem("Raw Transcript"):
                    output_raw = gr.Textbox(
                        interactive=False,
                        lines=20,
                        placeholder="Transcript will appear here (with speaker labels if diarization is on)..."
                    )

                with gr.TabItem("ℹ️ Model Card & Info"):
                    gr.Markdown("""
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
                
                    run_btn.click(
                        fn=process_meeting,
                        inputs=[
                            upload_input,
                            record_input,
                            lang_dropdown,
                            model_dropdown,
                            noise_red,
                            format_choice,
                            custom_vocab,
                            enable_diarization
                        ],
                        outputs=[output_raw, output_summary, download_output]
                    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,   # Set to False because share=True is not supported directly on HF Spaces
    debug=True,
    allowed_paths=[tempfile.gettempdir()]  
)
