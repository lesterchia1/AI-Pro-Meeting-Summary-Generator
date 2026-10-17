"""
Pro AI Meeting Minutes Assistant — v6.0 (Streamlit)
Whisper STT + multi-provider LLMs + SDLC action tracker (SQLite) + template ingestion + email drafts.
"""

import io
import json
import logging
import os
import re
import smtplib
import sqlite3
import tempfile
from datetime import datetime
from email.message import EmailMessage
from urllib.parse import quote

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pro AI Meeting Minutes", page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------------
# Compatibility shim: faster-whisper <= 1.2.x calls av.open(..., metadata_errors=...),
# an argument PyAV 19 removed. Drop it when the installed PyAV rejects it.
# ----------------------------------------------------------------------------
def _patch_pyav():
    try:
        import av

        if getattr(av.open, "_minutes_patched", False):
            return
        _orig_open = av.open

        def _open(*args, **kwargs):
            try:
                return _orig_open(*args, **kwargs)
            except TypeError as e:
                if "metadata_errors" in str(e) and "metadata_errors" in kwargs:
                    kwargs.pop("metadata_errors")
                    return _orig_open(*args, **kwargs)
                raise

        _open._minutes_patched = True
        av.open = _open
    except Exception:
        pass  # PyAV missing/odd: let faster-whisper raise its own clear error


_patch_pyav()

# ----------------------------------------------------------------------------
# Paths & logging
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "tracker_hub.db")
os.makedirs(LOG_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def setup_logging():
    logger = logging.getLogger("minutes_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s")
        fh = logging.FileHandler(
            os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log"), encoding="utf-8"
        )
        sh = logging.StreamHandler()
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


log = setup_logging()

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SDLC_PHASES = [
    "Planning", "Requirements gathering", "Design", "Implementation",
    "Testing", "Deployment", "Maintenance",
]
ACTION_STATES = ["Follow-up", "Pending Reply", "Accepted", "Rejected", "In Progress", "Completed"]
DEFAULT_PHASE, DEFAULT_STATE = "Planning", "Follow-up"

LANGUAGE_MAP = {"English": "en", "Malay": "ms", "Chinese": "zh", "Auto-Detect": None}
WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

SEALION_DEFAULT_URL = "https://api.sea-lion.ai/v1"

# Provider ids: groq / sealion use OpenAI-compatible endpoints; hf uses HF Inference (or local).
MODELS = {
    "Groq · GPT-OSS 120B": {"provider": "groq", "id": "openai/gpt-oss-120b", "max_chars": 300_000},
    "Groq · GPT-OSS 20B": {"provider": "groq", "id": "openai/gpt-oss-20b", "max_chars": 300_000},
    "AI Singapore · Sea-Lion v4 27B": {"provider": "sealion", "id": "aisingapore/Gemma-SEA-LION-v4-27B-IT", "max_chars": 200_000},
    "AI Singapore · Sea-Lion v4 32B (Qwen)": {"provider": "sealion", "id": "aisingapore/Qwen-SEA-LION-v4-32B-IT", "max_chars": 80_000},
    "Hugging Face · Qwen 3.5 2B": {"provider": "hf", "id": "Qwen/Qwen3.5-2B", "max_chars": 80_000},
}

MAX_TEMPLATE_CHARS = 6000


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets, then environment; never crash if secrets.toml is absent."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# ----------------------------------------------------------------------------
# SQLite tracker (Job Hunt Hub style)
# ----------------------------------------------------------------------------
TRACKER_COLUMNS = [
    "id", "meeting_date", "meeting_title", "task_description", "sdlc_phase",
    "status", "owner", "target_date", "notes", "model",
]


def db_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_date TEXT,
                meeting_title TEXT,
                task_description TEXT,
                sdlc_phase TEXT,
                status TEXT,
                owner TEXT,
                target_date TEXT,
                notes TEXT,
                model TEXT
            )
            """
        )
        # Lightweight migration for DBs created by the older Gradio version
        existing = {r[1] for r in conn.execute("PRAGMA table_info(action_items)")}
        for col in ("meeting_title", "model"):
            if col not in existing:
                conn.execute(f"ALTER TABLE action_items ADD COLUMN {col} TEXT")


def insert_actions(items, meeting_date, meeting_title, model_name):
    rows = [
        (
            meeting_date, meeting_title, it["task_description"], it["sdlc_phase"], it["status"],
            it["owner"], it["target_date"], it["notes"], model_name,
        )
        for it in items
    ]
    with db_conn() as conn:
        conn.executemany(
            "INSERT INTO action_items (meeting_date, meeting_title, task_description, sdlc_phase, "
            "status, owner, target_date, notes, model) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def load_tracker() -> pd.DataFrame:
    with db_conn() as conn:
        return pd.read_sql_query(
            f"SELECT {', '.join(TRACKER_COLUMNS)} FROM action_items ORDER BY id DESC", conn
        )


def update_tracker_rows(df: pd.DataFrame):
    with db_conn() as conn:
        for _, r in df.iterrows():
            conn.execute(
                "UPDATE action_items SET task_description=?, sdlc_phase=?, status=?, owner=?, "
                "target_date=?, notes=? WHERE id=?",
                (
                    r["task_description"], r["sdlc_phase"], r["status"], r["owner"],
                    r["target_date"], r["notes"], int(r["id"]),
                ),
            )


init_db()


# ----------------------------------------------------------------------------
# Speech-to-text
# ----------------------------------------------------------------------------
def detect_device():
    try:
        import ctranslate2  # installed as a dependency of faster-whisper

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper(size: str):
    from faster_whisper import WhisperModel

    device, compute_type = detect_device()
    log.info("Loading Whisper '%s' on %s (%s)", size, device, compute_type)
    return WhisperModel(size, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4)


def transcribe(path: str, size: str, lang_code, use_vad: bool, use_spectral_nr: bool):
    model = load_whisper(size)
    audio = path
    if use_spectral_nr:
        import noisereduce as nr
        from faster_whisper import decode_audio

        wav = decode_audio(path, sampling_rate=16000)
        audio = nr.reduce_noise(y=wav, sr=16000, stationary=True, prop_decrease=0.8).astype("float32")

    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language=lang_code,
        task="transcribe",
        vad_filter=use_vad,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,  # reduces repetition loops on noisy audio
    )
    segs = list(segments)
    lines = []
    for s in segs:
        m, sec = divmod(int(s.start), 60)
        lines.append(f"[{m:02d}:{sec:02d}] {s.text.strip()}")
    return "\n".join(lines), getattr(info, "language", None)


# ----------------------------------------------------------------------------
# Template ingestion
# ----------------------------------------------------------------------------
def extract_template_text(uploaded) -> str:
    if uploaded is None:
        return ""
    ext = os.path.splitext(uploaded.name)[1].lower()
    data = io.BytesIO(uploaded.getvalue())
    parts = []
    try:
        if ext == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(data)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        elif ext == ".docx":
            from docx import Document

            doc = Document(data)
            for p in doc.paragraphs:
                if p.text.strip():
                    style = p.style.name if p.style is not None else ""
                    prefix = "# " if style.lower().startswith("heading") else ""
                    parts.append(prefix + p.text)
            for t_idx, table in enumerate(doc.tables, 1):
                parts.append(f"[Table {t_idx}]")
                for row in table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
    except Exception as e:
        log.error("Template parse error: %s", e, exc_info=True)
        st.warning(f"Could not read the template ({e}). Continuing without it.")
        return ""
    text = "\n".join(parts).strip()
    if len(text) > MAX_TEMPLATE_CHARS:
        text = text[:MAX_TEMPLATE_CHARS]
        st.info(f"Template truncated to {MAX_TEMPLATE_CHARS} characters.")
    return text


# ----------------------------------------------------------------------------
# LLM calls
# ----------------------------------------------------------------------------
def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _openai_compatible(base_url, api_key, model_id, system, user, temperature):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


@st.cache_resource(show_spinner="Loading Qwen 3.5 2B locally…")
def _load_local_qwen(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # needs transformers>=5.2.0

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return tok, mdl


def _hf_call(model_id, system, user, temperature):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if get_secret("ENABLE_LOCAL_HF") == "1":
        tok, mdl = _load_local_qwen(model_id)
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(mdl.device)
        out = mdl.generate(**inputs, max_new_tokens=2048, do_sample=temperature > 0, temperature=max(temperature, 0.01))
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    token = get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set (add it to Streamlit secrets or environment).")
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token, timeout=180)
    resp = client.chat_completion(messages=messages, model=model_id, max_tokens=2048, temperature=temperature)
    return resp.choices[0].message.content or ""


def call_llm(model_name: str, system: str, user: str, temperature: float = 0.2) -> str:
    cfg = MODELS[model_name]
    try:
        if cfg["provider"] == "groq":
            key = get_secret("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            out = _openai_compatible("https://api.groq.com/openai/v1", key, cfg["id"], system, user, temperature)
        elif cfg["provider"] == "sealion":
            key = get_secret("SEALION_API_KEY")
            if not key:
                raise RuntimeError("SEALION_API_KEY is not set.")
            out = _openai_compatible(
                get_secret("SEALION_BASE_URL", SEALION_DEFAULT_URL), key, cfg["id"], system, user, temperature
            )
        else:
            out = _hf_call(cfg["id"], system, user, temperature)
        return strip_reasoning(out)
    except Exception as e:
        log.error("LLM error (%s): %s", model_name, e, exc_info=True)
        raise RuntimeError(f"{model_name}: {e}") from e


SYSTEM_PROMPT = (
    "You are an expert Chief of Staff and technical project manager. Use ONLY information present in the "
    "transcript. If an owner, date, or detail was not stated, write 'Not stated' instead of guessing. "
    "Never invent attendees, decisions, or numbers."
)


def build_minutes_prompt(transcript, lang_name, template_text):
    if template_text:
        layout = (
            "Format the minutes to match the layout of this user-provided template exactly "
            "(same section order, headings and table structure). Fill each section only from the transcript; "
            "write 'Not stated' where the transcript has no information.\n"
            f"--- TEMPLATE START ---\n{template_text}\n--- TEMPLATE END ---\n"
        )
    else:
        layout = (
            "Use these sections: 1. Executive Overview; 2. Key Decisions; 3. Action Items (owner, target date, "
            f"SDLC phase from [{', '.join(SDLC_PHASES)}], status from [{', '.join(ACTION_STATES)}]); "
            "4. Open Risks / Blockers.\n"
        )
    return (
        f"Write meeting minutes in {lang_name if lang_name != 'Auto-Detect' else 'the language of the transcript'}.\n"
        f"{layout}\nTranscript:\n{transcript}"
    )


def build_actions_prompt(transcript, minutes):
    return (
        "Extract action items from the meeting below. Return ONLY a JSON array (no prose, no code fences). "
        "Each element must have exactly these keys: "
        '"task_description", "owner", "target_date", "sdlc_phase", "status", "notes".\n'
        f"sdlc_phase must be one of {SDLC_PHASES}. status must be one of {ACTION_STATES}. "
        'Use "Not stated" for unknown owner/target_date. Only include tasks actually mentioned. '
        "If there are none, return [].\n\n"
        f"Minutes:\n{minutes}\n\nTranscript:\n{transcript}"
    )


def parse_actions(raw: str):
    """Parse and validate the model's JSON; returns a list of normalised dicts (possibly empty)."""
    text = re.sub(r"```(?:json)?", "", raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Model output was not a JSON array.")

    def pick(value, allowed, default):
        v = str(value or "").strip().lower()
        for a in allowed:
            if v == a.lower() or v.startswith(a.lower().split()[0]):
                return a
        return default

    items = []
    for d in data:
        if not isinstance(d, dict) or not str(d.get("task_description", "")).strip():
            continue
        items.append({
            "task_description": str(d["task_description"]).strip(),
            "owner": str(d.get("owner") or "Not stated").strip(),
            "target_date": str(d.get("target_date") or "Not stated").strip(),
            "sdlc_phase": pick(d.get("sdlc_phase"), SDLC_PHASES, DEFAULT_PHASE),
            "status": pick(d.get("status"), ACTION_STATES, DEFAULT_STATE),
            "notes": str(d.get("notes") or "").strip(),
        })
    return items


# ----------------------------------------------------------------------------
# Email helpers
# ----------------------------------------------------------------------------
def smtp_configured() -> bool:
    return all(get_secret(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def send_email_smtp(to_addrs: str, subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = get_secret("SMTP_FROM") or get_secret("SMTP_USER")
    msg["To"] = to_addrs
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(get_secret("SMTP_HOST"), int(get_secret("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls()
        s.login(get_secret("SMTP_USER"), get_secret("SMTP_PASSWORD"))
        s.send_message(msg)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def save_to_temp(uploaded, default_suffix=".wav") -> str:
    suffix = os.path.splitext(getattr(uploaded, "name", "") or "")[1] or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def run_pipeline(audio_file, template_file, title, lang_name, model_name, whisper_size, use_vad, use_nr):
    tmp_path = save_to_temp(audio_file)
    meeting_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with st.status("Processing meeting…", expanded=True) as status:
            st.write(f"🎧 Transcribing ({lang_name}, Whisper {whisper_size})…")
            transcript, detected = transcribe(tmp_path, whisper_size, LANGUAGE_MAP[lang_name], use_vad, use_nr)
            if not transcript.strip():
                status.update(label="No speech detected", state="error")
                st.error("Transcript is empty — silence or very low audio quality.")
                return

            cap = MODELS[model_name]["max_chars"]
            llm_transcript = transcript
            if len(transcript) > cap:
                llm_transcript = transcript[:cap]
                st.warning(f"Transcript exceeds this model's safe input size; the first {cap:,} characters were sent.")

            st.write("📑 Generating minutes…")
            tpl_text = extract_template_text(template_file)
            minutes = call_llm(model_name, SYSTEM_PROMPT, build_minutes_prompt(llm_transcript, lang_name, tpl_text))

            st.write("🗂️ Extracting action items…")
            inserted, parse_err = 0, None
            try:
                raw = call_llm(model_name, SYSTEM_PROMPT, build_actions_prompt(llm_transcript, minutes), 0.0)
                inserted = insert_actions(parse_actions(raw), meeting_date, title, model_name)
            except Exception as e:  # keep the minutes even if extraction fails
                parse_err = str(e)
                log.warning("Action extraction failed: %s", e)

            st.session_state.update(
                transcript=transcript, minutes=minutes, title=title, email_draft="", detected=detected
            )
            if parse_err:
                st.warning(f"Minutes generated, but action items could not be parsed: {parse_err}")
            status.update(label=f"Done — {inserted} action item(s) logged", state="complete")
    except Exception as e:
        log.error("Pipeline error: %s", e, exc_info=True)
        st.error(f"Processing failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🎙️ Pro AI Meeting Minutes Assistant")
st.caption("v6.0 · Whisper STT · SDLC tracker · template ingestion · email drafts")

with st.sidebar:
    st.header("Configuration")
    lang_name = st.selectbox("Speaking language", list(LANGUAGE_MAP.keys()), index=0)
    model_name = st.selectbox("AI engine", list(MODELS.keys()), index=0)
    whisper_size = st.selectbox(
        "Whisper model", WHISPER_SIZES, index=2,
        help="Larger = more accurate in noise but slower and needs more RAM. Use small or below on Streamlit Cloud.",
    )
    use_vad = st.checkbox("Voice-activity detection (skip silence/noise)", value=True)
    use_nr = st.checkbox("Spectral noise reduction (noisereduce)", value=False,
                         help="Stationary background-noise reduction before transcription. Adds processing time.")
    st.divider()
    st.markdown("**API keys detected**")
    st.write(f"Groq: {'✅' if get_secret('GROQ_API_KEY') else '❌'}  ·  "
             f"Sea-Lion: {'✅' if get_secret('SEALION_API_KEY') else '❌'}  ·  "
             f"HF: {'✅' if get_secret('HF_TOKEN') else '❌'}")

col_in, col_tpl = st.columns(2)
with col_in:
    uploaded_audio = st.file_uploader(
        "Upload audio / video", type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "webm", "mp4", "mov", "mkv"]
    )
    recorded_audio = st.audio_input("…or record live audio") if hasattr(st, "audio_input") else None
with col_tpl:
    template_file = st.file_uploader("Custom meeting template (PDF / DOCX, optional)", type=["pdf", "docx"])
    meeting_title = st.text_input("Meeting title", value="Untitled meeting")

if st.button("🚀 Process meeting & update tracker", type="primary"):
    source = uploaded_audio or recorded_audio
    if source is None:
        st.warning("Please upload or record audio first.")
    else:
        run_pipeline(source, template_file, meeting_title.strip() or "Untitled meeting",
                     lang_name, model_name, whisper_size, use_vad, use_nr)

tab_min, tab_email, tab_track, tab_raw = st.tabs(
    ["📝 Minutes", "✉️ Email", "📊 Job Hunt Hub & SDLC Tracker", "🗒️ Raw transcript"]
)

with tab_min:
    minutes = st.session_state.get("minutes")
    if minutes:
        st.markdown(minutes)
        st.download_button("Download minutes (.md)", minutes, file_name="meeting_minutes.md")
    else:
        st.info("Minutes will appear here after processing.")

with tab_email:
    minutes = st.session_state.get("minutes")
    if not minutes:
        st.info("Process a meeting first, then draft a follow-up email here.")
    else:
        to_addrs = st.text_input("Recipients (comma-separated)", key="email_to")
        subject = st.text_input("Subject", value=f"Meeting minutes: {st.session_state.get('title', '')}", key="email_subject")
        tone = st.selectbox("Tone", ["Professional", "Concise", "Friendly"], key="email_tone")
        if st.button("Draft email with AI"):
            try:
                with st.spinner("Drafting…"):
                    st.session_state["email_draft"] = call_llm(
                        model_name, SYSTEM_PROMPT,
                        f"Write a {tone.lower()} follow-up email (body only, no subject line) summarising these "
                        f"meeting minutes: key decisions and action items with owners and dates. Do not add facts "
                        f"not in the minutes.\n\n{minutes}",
                    )
            except Exception as e:
                st.error(str(e))
        draft = st.text_area("Email body (editable)", value=st.session_state.get("email_draft", ""), height=300, key="email_body")
        if draft:
            mailto = f"mailto:{quote(to_addrs)}?subject={quote(subject)}&body={quote(draft)}"
            st.link_button("Open in my email app", mailto)
            st.caption("Very long drafts may be truncated by some mail clients; copy/paste if so.")
            if smtp_configured():
                if st.button("Send via SMTP"):
                    try:
                        send_email_smtp(to_addrs, subject, draft)
                        st.success("Email sent.")
                    except Exception as e:
                        log.error("SMTP error: %s", e, exc_info=True)
                        st.error(f"Send failed: {e}")
            else:
                st.caption("Add SMTP_* secrets to enable direct sending.")

with tab_track:
    df = load_tracker()
    if df.empty:
        st.info("No action items yet.")
    else:
        f1, f2, f3 = st.columns(3)
        phase_f = f1.multiselect("SDLC phase", SDLC_PHASES)
        status_f = f2.multiselect("Status", ACTION_STATES)
        text_f = f3.text_input("Search task / owner / meeting")
        view = df
        if phase_f:
            view = view[view["sdlc_phase"].isin(phase_f)]
        if status_f:
            view = view[view["status"].isin(status_f)]
        if text_f:
            mask = (
                view["task_description"].fillna("").str.contains(text_f, case=False)
                | view["owner"].fillna("").str.contains(text_f, case=False)
                | view["meeting_title"].fillna("").str.contains(text_f, case=False)
            )
            view = view[mask]

        edited = st.data_editor(
            view,
            hide_index=True,
            use_container_width=True,
            disabled=["id", "meeting_date", "meeting_title", "model"],
            column_config={
                "sdlc_phase": st.column_config.SelectboxColumn("SDLC phase", options=SDLC_PHASES, required=True),
                "status": st.column_config.SelectboxColumn("Status", options=ACTION_STATES, required=True),
            },
            key="tracker_editor",
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("💾 Save edits"):
            update_tracker_rows(edited)
            st.success("Tracker updated.")
            st.rerun()
        c2.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), "tracker_hub.csv", "text/csv")
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        c3.download_button(
            "Export Excel", buf.getvalue(), "tracker_hub.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with tab_raw:
    transcript = st.session_state.get("transcript")
    if transcript:
        if st.session_state.get("detected"):
            st.caption(f"Detected language: {st.session_state['detected']}")
        st.text_area("Transcript", transcript, height=420)
        st.download_button("Download transcript (.txt)", transcript, file_name="transcript.txt")
    else:
        st.info("Transcript will appear here after processing.")
