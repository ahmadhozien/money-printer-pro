import contextlib
import io
import json
import os
import re
import sys
import traceback
from uuid import uuid4
from datetime import datetime
from html import escape

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cache import add_account, get_accounts, remove_account, update_account
from classes.Tts import TTS
from classes.TikTok import TikTok
from classes.Trends import get_trending_ideas, language_to_code, list_trend_categories
from classes.Twitter import Twitter
from classes.YouTube import ImageRateLimitError, YouTube
from config import ROOT_DIR as APP_ROOT_DIR
from config import get_configured_llm_model, get_is_for_kids, get_llm_provider, get_pricing_config
from llm_provider import list_models, select_model

CONFIG_PATH = os.path.join(APP_ROOT_DIR, "config.json")
YOUTUBE_DRAFTS_PATH = os.path.join(APP_ROOT_DIR, ".mp", "youtube_drafts.json")
YOUTUBE_RUNS_PATH = os.path.join(APP_ROOT_DIR, ".mp", "youtube_runs")
YOUTUBE_PIXABAY_PICKER_STATE_KEY = "youtube_pixabay_picker"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def _send_trend_to_seed(account_id: str, title: str) -> None:
    """
    Button callback: apply a chosen trending idea to the Create tab. Sets both
    the title/keyword seed and the Niche channel-override so the generated video
    is built around the trend. Runs before widgets re-instantiate, so setting
    the widget keys here is safe.
    """
    st.session_state[f"youtube_seed_topic_{account_id}"] = title
    st.session_state[f"yt_niche_{account_id}"] = title
    st.session_state["_trend_seed_notice"] = title


def load_youtube_drafts() -> dict:
    if not os.path.exists(YOUTUBE_DRAFTS_PATH):
        return {"drafts": []}

    try:
        with open(YOUTUBE_DRAFTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"drafts": []}


def save_youtube_drafts(payload: dict) -> None:
    with open(YOUTUBE_DRAFTS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def infer_youtube_workspace_dir(video_path: str = "", workspace_dir: str = "") -> str:
    normalized_workspace = os.path.abspath(str(workspace_dir or "").strip()) if str(workspace_dir or "").strip() else ""
    if normalized_workspace and os.path.exists(os.path.join(normalized_workspace, "run_state.json")):
        return normalized_workspace

    normalized_video_path = os.path.abspath(str(video_path or "").strip()) if str(video_path or "").strip() else ""
    if not normalized_video_path:
        return ""

    candidate = os.path.dirname(normalized_video_path)
    if os.path.exists(os.path.join(candidate, "run_state.json")):
        return candidate
    return ""


def build_youtube_editor_payload(source_payload: dict | None = None) -> dict:
    payload = dict(source_payload or {})
    workspace_dir = infer_youtube_workspace_dir(
        video_path=str(payload.get("video_path", "") or ""),
        workspace_dir=str(payload.get("workspace_dir", "") or ""),
    )
    workspace_payload = load_youtube_run_state(workspace_dir) if workspace_dir else None
    merged = dict(workspace_payload or {})
    merged.update(payload)
    if workspace_dir:
        merged["workspace_dir"] = workspace_dir

    def resolve_workspace_path(raw_path: str) -> str:
        cleaned_path = str(raw_path or "").strip()
        if not cleaned_path:
            return ""
        if os.path.isabs(cleaned_path):
            return os.path.abspath(cleaned_path)
        if workspace_dir:
            return os.path.abspath(os.path.join(workspace_dir, cleaned_path))
        return os.path.abspath(cleaned_path)

    script_text = str(merged.get("script", "") or "").strip()
    scene_units = [
        str(item).strip()
        for item in list(merged.get("scene_units", []) or [])
        if str(item).strip()
    ]
    image_prompts = [
        str(item).strip()
        for item in list(merged.get("image_prompts", []) or [])
        if str(item).strip()
    ]

    if not scene_units and script_text:
        scene_units = [
            part.strip(" \t\r\n-")
            for part in re.split(r"(?<=[\.\!\?\u061f])\s+|\n+", script_text)
            if part and part.strip()
        ]

    if not scene_units and image_prompts:
        scene_units = [prompt for prompt in image_prompts]
    if not image_prompts and scene_units:
        image_prompts = [scene for scene in scene_units]

    target_count = max(len(scene_units), len(image_prompts))
    if target_count and len(scene_units) < target_count:
        scene_units.extend(scene_units[-1:] * (target_count - len(scene_units)))
    if target_count and len(image_prompts) < target_count:
        fallback_prompt = image_prompts[-1] if image_prompts else scene_units[-1]
        image_prompts.extend([fallback_prompt] * (target_count - len(image_prompts)))

    visual_assets = {
        int(asset.get("scene_index", -1)): asset
        for asset in list(merged.get("visual_assets", []) or [])
        if isinstance(asset, dict) and isinstance(asset.get("scene_index"), int)
    }
    scenes = []
    for index in range(target_count):
        asset = visual_assets.get(index, {})
        scenes.append(
            {
                "scene_text": scene_units[index] if index < len(scene_units) else "",
                "image_prompt": image_prompts[index] if index < len(image_prompts) else "",
                "asset_path": resolve_workspace_path(str(asset.get("path", "") or "")),
                "asset_type": str(asset.get("type", "") or ""),
                "asset_source": str(asset.get("source", "") or ""),
                "pixabay_query": str(asset.get("pixabay_query", "") or ""),
                "pixabay_query_type": str(asset.get("pixabay_query_type", "") or ""),
                "scene_index": index,
            }
        )

    merged["metadata"] = dict(merged.get("metadata", {}) or {})
    merged["subject"] = str(merged.get("subject", "") or "").strip()
    merged["script"] = script_text or "\n\n".join(scene_units)
    merged["video_path"] = resolve_workspace_path(str(merged.get("video_path", "") or ""))
    merged["tts_path"] = resolve_workspace_path(str(merged.get("tts_path", "") or ""))
    merged["scenes"] = scenes
    merged["scene_units"] = scene_units[:target_count]
    merged["image_prompts"] = image_prompts[:target_count]
    return merged


def save_youtube_draft(
    account_id: str,
    video_path: str,
    metadata: dict,
    workspace_dir: str = "",
    subject: str = "",
    script: str = "",
    scene_units: list[str] | None = None,
    image_prompts: list[str] | None = None,
) -> None:
    drafts_payload = load_youtube_drafts()
    drafts = drafts_payload.get("drafts", [])
    normalized_path = os.path.abspath(video_path)
    normalized_metadata = dict(metadata or {})
    pricing_payload = load_pricing_for_video(normalized_path, normalized_metadata)
    if pricing_payload:
        normalized_metadata["pricing"] = pricing_payload
    normalized_workspace_dir = infer_youtube_workspace_dir(normalized_path, workspace_dir)
    workspace_payload = load_youtube_run_state(normalized_workspace_dir) if normalized_workspace_dir else None

    drafts = [
        draft for draft in drafts
        if not (
            draft.get("account_id") == account_id
            and os.path.abspath(draft.get("video_path", "")) == normalized_path
        )
    ]
    drafts.append(
        {
            "account_id": account_id,
            "video_path": normalized_path,
            "metadata": normalized_metadata,
            "workspace_dir": normalized_workspace_dir,
            "subject": str(
                subject or (workspace_payload.get("subject", "") if workspace_payload else "")
            ).strip(),
            "script": str(
                script or (workspace_payload.get("script", "") if workspace_payload else "")
            ).strip(),
            "scene_units": list(
                scene_units
                if scene_units is not None
                else (workspace_payload.get("scene_units", []) if workspace_payload else [])
            ),
            "image_prompts": list(
                image_prompts
                if image_prompts is not None
                else (workspace_payload.get("image_prompts", []) if workspace_payload else [])
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    drafts_payload["drafts"] = drafts[-20:]
    save_youtube_drafts(drafts_payload)


def get_youtube_drafts_for_account(account_id: str) -> list[dict]:
    drafts_payload = load_youtube_drafts()
    drafts = [
        draft for draft in drafts_payload.get("drafts", [])
        if draft.get("account_id") == account_id and os.path.exists(draft.get("video_path", ""))
    ]
    return sorted(drafts, key=lambda draft: draft.get("created_at", ""), reverse=True)


def get_last_youtube_draft(account_id: str) -> dict | None:
    drafts = get_youtube_drafts_for_account(account_id)
    return drafts[0] if drafts else None


def get_all_youtube_drafts() -> list[dict]:
    drafts_payload = load_youtube_drafts()
    drafts = [
        draft for draft in drafts_payload.get("drafts", [])
        if os.path.exists(draft.get("video_path", ""))
    ]
    return sorted(drafts, key=lambda draft: draft.get("created_at", ""), reverse=True)


def load_youtube_run_state(workspace_dir: str) -> dict | None:
    state_path = os.path.join(os.path.abspath(workspace_dir), "run_state.json")
    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as file:
            payload = json.load(file) or {}
    except (OSError, json.JSONDecodeError):
        return None

    payload["workspace_dir"] = os.path.abspath(workspace_dir)
    return payload


def get_youtube_recoverable_runs_for_account(account_id: str) -> list[dict]:
    if not os.path.isdir(YOUTUBE_RUNS_PATH):
        return []

    recoverable_runs = []
    for entry in os.scandir(YOUTUBE_RUNS_PATH):
        if not entry.is_dir():
            continue
        payload = load_youtube_run_state(entry.path)
        if not payload or payload.get("account_uuid") != account_id:
            continue

        workspace_dir = payload["workspace_dir"]
        voiceover_path = payload.get("tts_path", "")
        if voiceover_path and not os.path.isabs(voiceover_path):
            voiceover_path = os.path.join(workspace_dir, voiceover_path)
        visual_assets = payload.get("visual_assets", [])
        pricing_path = os.path.join(workspace_dir, "pricing.json")
        final_video_path = payload.get("video_path", "") or os.path.join(workspace_dir, "final_video.mp4")
        if final_video_path and not os.path.isabs(final_video_path):
            final_video_path = os.path.join(workspace_dir, final_video_path)

        if not os.path.exists(str(voiceover_path or "")):
            continue
        if not isinstance(visual_assets, list) or len(visual_assets) == 0:
            continue
        if os.path.exists(pricing_path) and os.path.exists(final_video_path):
            continue

        payload["voiceover_path"] = os.path.abspath(voiceover_path)
        payload["final_video_path"] = os.path.abspath(final_video_path)
        payload["visual_asset_count"] = len(visual_assets)
        recoverable_runs.append(payload)

    return sorted(
        recoverable_runs,
        key=lambda item: str(item.get("updated_at", "")),
        reverse=True,
    )


def format_recoverable_run_option(run_payload: dict) -> str:
    subject = str(run_payload.get("subject", "") or "(untitled)").strip()
    updated_at = str(run_payload.get("updated_at", "") or "").replace("T", " ")
    return (
        f"{updated_at or 'Unfinished run'} | "
        f"{subject[:70]} | "
        f"{int(run_payload.get('visual_asset_count', 0) or 0)} assets"
    )


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MOVIEPY_PROGRESS_RE = re.compile(r"^(?:[\w-]+:\s*)?\d{1,3}%\|")


class StreamlitLogCapture(io.TextIOBase):
    def __init__(self, placeholder, max_lines: int = 120):
        self.placeholder = placeholder
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.pending = ""

    def write(self, value):
        text = ANSI_ESCAPE_RE.sub("", str(value or "")).replace("\r", "\n")
        if not text:
            return 0

        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            cleaned = line.strip()
            if cleaned and not MOVIEPY_PROGRESS_RE.match(cleaned):
                self.lines.append(cleaned)

        self.lines = self.lines[-self.max_lines:]
        self._render()
        return len(text)

    def flush(self):
        cleaned = self.pending.strip()
        if cleaned and not MOVIEPY_PROGRESS_RE.match(cleaned):
            self.lines.append(cleaned)
            self.lines = self.lines[-self.max_lines:]
            self.pending = ""
            self._render()

    def append(self, message: str) -> None:
        cleaned = ANSI_ESCAPE_RE.sub("", str(message or "")).strip()
        if not cleaned:
            return
        self.lines.append(cleaned)
        self.lines = self.lines[-self.max_lines:]
        self._render()

    def _render(self):
        if self.placeholder is not None:
            self.placeholder.code("\n".join(self.lines[-80:]), language="text")


def build_youtube_generation_monitor():
    # ── Progress bar + step label (always visible at top) ───────────
    progress_caption = st.empty()
    progress_bar = st.progress(0)

    # ── Two-column layout: left = live feed, right = content preview ─
    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        status_placeholder = st.empty()
        assets_placeholder = st.empty()
    with right_col:
        subject_placeholder = st.empty()
        metadata_summary_placeholder = st.empty()
        script_placeholder = st.empty()

    # ── Collapsible details ─────────────────────────────────────────
    with st.expander("Image prompts", expanded=False):
        prompts_placeholder = st.empty()
    with st.expander("Technical log", expanded=False):
        logs_placeholder = st.empty()

    log_capture = StreamlitLogCapture(logs_placeholder)
    seen_assets: list[str] = []
    progress_state = {
        "value": 0,
        "expected_assets": 0,
        "completed_assets": 0,
    }

    def set_progress(value: int, label: str | None = None) -> None:
        bounded_value = max(0, min(100, int(value)))
        progress_state["value"] = max(progress_state["value"], bounded_value)
        progress_bar.progress(progress_state["value"])
        if label:
            progress_caption.caption(f"{progress_state['value']}% — {label}")

    def update_assets_progress() -> None:
        expected = int(progress_state.get("expected_assets", 0) or 0)
        completed = int(progress_state.get("completed_assets", 0) or 0)
        if expected <= 0:
            set_progress(50, "Preparing visual assets...")
            return
        ratio = min(1.0, completed / max(expected, 1))
        set_progress(45 + int(ratio * 25), f"Assets {completed}/{expected}")

    def render_compact_note(title: str, body: str) -> str:
        return (
            f'<div style="padding:0.6rem 0.8rem;border-radius:14px;'
            f'background:rgba(208,240,215,0.35);border:1px solid rgba(31,122,90,0.08);'
            f'margin-bottom:0.5rem;font-size:0.88rem;line-height:1.55;">'
            f'<strong style="font-size:0.76rem;text-transform:uppercase;letter-spacing:0.06em;'
            f'color:#3a6b55;">{escape(title)}</strong><br>'
            f'<div style="white-space:pre-wrap;color:#224739;" dir="auto">{escape(body)}</div>'
            f'</div>'
        )

    def progress_callback(event: dict) -> None:
        stage = str((event or {}).get("stage", "")).strip().lower()
        message = str((event or {}).get("message", "")).strip()
        payload = (event or {}).get("payload", {}) or {}

        if message:
            status_placeholder.markdown(
                render_compact_note("Current Step", message),
                unsafe_allow_html=True,
            )
            if stage != "render_progress":
                log_capture.append(message)

        stage_progress = {
            "topic": 10, "script": 20, "metadata_start": 25, "metadata": 30,
            "prompts_start": 35, "image_prompts": 40, "assets_start": 45,
            "tts_start": 78, "tts": 82, "recovery": 84, "combine": 86,
            "subtitles": 90, "subtitles_start": 88, "render_start": 92,
            "render": 94, "done": 100,
        }
        if stage in stage_progress:
            set_progress(stage_progress[stage], message or stage.replace("_", " ").title())

        if payload.get("subject"):
            subject_placeholder.markdown(
                render_compact_note("Topic", str(payload["subject"]).strip()),
                unsafe_allow_html=True,
            )

        if payload.get("script"):
            script_text = str(payload["script"]).strip()
            # Show first 200 chars with ellipsis to keep it compact
            preview = script_text[:200] + ("..." if len(script_text) > 200 else "")
            script_placeholder.markdown(
                render_compact_note("Script", preview),
                unsafe_allow_html=True,
            )

        if payload.get("metadata"):
            meta = payload["metadata"]
            title_str = str(meta.get("title", "")).strip()
            desc_str = str(meta.get("description", "")).strip()[:120]
            tags = meta.get("hashtags", [])[:5]
            summary = f"{title_str}\n{desc_str}"
            if tags:
                summary += "\n" + " ".join(str(t) for t in tags)
            metadata_summary_placeholder.markdown(
                render_compact_note("Metadata", summary),
                unsafe_allow_html=True,
            )

        if payload.get("image_prompts"):
            progress_state["expected_assets"] = len(payload["image_prompts"])
            progress_state["completed_assets"] = 0
            prompts_text = "\n".join(
                f"{i + 1}. {p[:90]}{'...' if len(p) > 90 else ''}"
                for i, p in enumerate(payload["image_prompts"])
            )
            prompts_placeholder.markdown(
                render_compact_note(f"Image Prompts ({len(payload['image_prompts'])})", prompts_text),
                unsafe_allow_html=True,
            )

        if stage == "visual_asset":
            progress_state["completed_assets"] = int(progress_state.get("completed_assets", 0) or 0) + 1
            update_assets_progress()
            scene_index = payload.get("scene_index")
            scene_label = f"S{int(scene_index) + 1}" if isinstance(scene_index, int) else "?"
            source = str(payload.get("source", "")).strip()
            atype = str(payload.get("asset_type", "")).strip()
            seen_assets.append(f"{scene_label}: {source} {atype}")
            # Show as compact inline chips
            chips = " ".join(
                f'<span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;'
                f'background:rgba(31,122,90,0.08);border:1px solid rgba(31,122,90,0.1);'
                f'font-size:0.78rem;margin:2px;">{escape(a)}</span>'
                for a in seen_assets[-20:]
            )
            assets_placeholder.markdown(
                f'<div style="padding:0.6rem 0.8rem;border-radius:14px;'
                f'background:rgba(208,240,215,0.35);border:1px solid rgba(31,122,90,0.08);'
                f'margin-bottom:0.5rem;">'
                f'<strong style="font-size:0.76rem;text-transform:uppercase;letter-spacing:0.06em;'
                f'color:#3a6b55;">Assets ({len(seen_assets)})</strong><br>'
                f'<div style="margin-top:0.3rem;">{chips}</div></div>',
                unsafe_allow_html=True,
            )

        if stage == "render_progress":
            render_ratio = max(0.0, min(1.0, float(payload.get("progress", 0.0) or 0.0)))
            set_progress(95 + int(render_ratio * 5), message or "Rendering final video...")

        if stage == "done" and payload.get("video_path"):
            set_progress(100, "Done!")

    return progress_callback, log_capture


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --studio-ink: #12211b;
            --studio-ink-soft: #41584e;
            --studio-accent: #1f7a5a;
            --studio-accent-2: #d0f0d7;
            --studio-warm: #f6f4ee;
            --studio-card: rgba(255, 255, 255, 0.92);
            --studio-border: rgba(18, 33, 27, 0.08);
            --studio-shadow: 0 18px 50px rgba(16, 24, 40, 0.08);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(214, 239, 224, 0.6), transparent 32%),
                radial-gradient(circle at top right, rgba(247, 220, 190, 0.45), transparent 24%),
                linear-gradient(180deg, #fbfaf6 0%, #f4f8f4 48%, #f8faf8 100%);
            color: var(--studio-ink);
            font-family: "Segoe UI Variable Text", "Aptos", "Trebuchet MS", sans-serif;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1.4rem;
        }
        .block-container {
            max-width: 1380px;
            padding-top: 1rem;
            padding-bottom: 4rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        [data-testid="stForm"] {
            border: 1px solid var(--studio-border);
            background: rgba(255, 255, 255, 0.78);
            border-radius: 22px;
            padding: 1.1rem 1rem 1.3rem;
            box-shadow: var(--studio-shadow);
        }
        [data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            border: 1px solid rgba(18, 33, 27, 0.08);
            padding: 0.65rem 1rem;
            background: rgba(255, 255, 255, 0.7);
            color: var(--studio-ink-soft);
            font-weight: 600;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #1f7a5a, #2d9d71);
            color: white;
            border-color: transparent;
            box-shadow: 0 10px 24px rgba(31, 122, 90, 0.22);
        }
        [data-testid="stTabs"] {
            margin-top: 1.15rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--studio-border);
            background: var(--studio-card);
            border-radius: 22px;
            padding: 0.55rem 0.9rem;
            box-shadow: var(--studio-shadow);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        [data-testid="stMetricValue"] {
            color: var(--studio-ink);
        }
        .stButton > button, .stDownloadButton > button, [data-testid="baseButton-secondary"] {
            border-radius: 999px;
            min-height: 2.9rem;
            border: 1px solid rgba(18, 33, 27, 0.08);
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1f7a5a, #2d9d71);
            color: white;
            border-color: transparent;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div, .stNumberInput input {
            border-radius: 16px !important;
            border-color: rgba(18, 33, 27, 0.12) !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }
        .stTextArea textarea {
            line-height: 1.65;
        }
        .stCheckbox, .stRadio {
            padding-top: 0.2rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--studio-border);
            background: rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            overflow: hidden;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.65rem 1.8rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(18, 33, 27, 0.08);
            background:
                radial-gradient(circle at right top, rgba(255,255,255,0.42), transparent 22%),
                linear-gradient(135deg, rgba(19, 62, 49, 0.96), rgba(34, 112, 82, 0.94));
            box-shadow: 0 24px 60px rgba(16, 24, 40, 0.14);
            color: #f5fff8;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -10% -38% auto;
            width: 360px;
            height: 360px;
            background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 62%);
            pointer-events: none;
        }
        .hero-eyebrow {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.78;
        }
        .hero-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0.45rem 0 0.75rem;
        }
        .hero-copy {
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.72;
            color: rgba(245, 255, 248, 0.92);
        }
        .hero-chip-row, .studio-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .hero-chip {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.14);
            color: #f3fff8;
            font-size: 0.82rem;
        }
        .section-intro {
            margin: 0.2rem 0 1rem;
            padding: 1rem 1.1rem 0.6rem;
            border-left: 4px solid rgba(31, 122, 90, 0.28);
        }
        .section-label {
            font-size: 0.74rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            font-weight: 700;
            color: #668072;
            margin-bottom: 0.35rem;
        }
        .section-title {
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 800;
            color: var(--studio-ink);
            margin: 0 0 0.45rem;
        }
        .section-copy {
            font-size: 0.98rem;
            color: var(--studio-ink-soft);
            line-height: 1.75;
            max-width: 880px;
        }
        .studio-card {
            border: 1px solid var(--studio-border);
            border-radius: 22px;
            padding: 18px 20px;
            background: linear-gradient(180deg, rgba(252, 253, 251, 0.96), rgba(255, 255, 255, 0.98));
            box-shadow: var(--studio-shadow);
            margin-bottom: 14px;
        }
        .studio-eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #668072;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }
        .studio-title {
            font-size: 1.18rem;
            line-height: 1.5;
            font-weight: 700;
            color: var(--studio-ink);
            margin-bottom: 0.55rem;
        }
        .studio-subtitle {
            font-size: 0.92rem;
            color: var(--studio-ink-soft);
            line-height: 1.6;
        }
        .studio-chip {
            display: inline-block;
            padding: 0.32rem 0.68rem;
            border-radius: 999px;
            background: rgba(31, 122, 90, 0.08);
            color: #26463b;
            font-size: 0.85rem;
            border: 1px solid rgba(31, 122, 90, 0.12);
        }
        .studio-chip--muted {
            background: #f8faf8;
            border-color: rgba(102, 128, 114, 0.16);
        }
        .studio-path {
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            background: #f8faf8;
            border: 1px solid rgba(102, 128, 114, 0.14);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            color: var(--studio-ink);
            word-break: break-all;
        }
        .studio-kv {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 0.8rem;
        }
        .studio-kv-item {
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
            background: rgba(244, 248, 244, 0.95);
            border: 1px solid rgba(102, 128, 114, 0.12);
        }
        .studio-kv-label {
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #6a8076;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .studio-kv-value {
            color: var(--studio-ink);
            font-size: 0.95rem;
            line-height: 1.5;
            font-weight: 600;
        }
        .studio-note {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(208, 240, 215, 0.44);
            border: 1px solid rgba(31, 122, 90, 0.1);
            color: #224739;
            font-size: 0.95rem;
            line-height: 1.65;
            margin-bottom: 0.9rem;
        }
        .studio-panel {
            border: 1px solid var(--studio-border);
            border-radius: 24px;
            padding: 1rem 1.1rem 1.15rem;
            background: rgba(255, 255, 255, 0.84);
            box-shadow: var(--studio-shadow);
            margin-bottom: 0.9rem;
        }
        .studio-panel--soft {
            background: rgba(249, 251, 248, 0.92);
        }
        .studio-panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--studio-ink);
            margin: 0 0 0.8rem;
        }
        .studio-panel-copy {
            font-size: 0.92rem;
            color: var(--studio-ink-soft);
            line-height: 1.65;
            margin-bottom: 0.8rem;
        }
        .studio-inline-actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .studio-mini-note {
            font-size: 0.84rem;
            color: #6a8076;
            line-height: 1.55;
        }
        .studio-divider-space {
            height: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, body: str, eyebrow: str = "Workspace") -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-label">{escape(eyebrow)}</div>
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, body: str, chips: list[str] | None = None, eyebrow: str = "Studio") -> None:
    chip_markup = ""
    cleaned_chips = [chip.strip() for chip in (chips or []) if str(chip).strip()]
    if cleaned_chips:
        chip_markup = '<div class="hero-chip-row">' + "".join(
            f'<span class="hero-chip">{escape(chip)}</span>' for chip in cleaned_chips
        ) + "</div>"

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-eyebrow">{escape(eyebrow)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-copy">{escape(body)}</div>
            {chip_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kv_card(eyebrow: str, title: str, fields: list[tuple[str, str]]) -> None:
    items = "".join(
        f"""
        <div class="studio-kv-item">
            <div class="studio-kv-label">{escape(label)}</div>
            <div class="studio-kv-value" dir="auto">{escape(value or "Not set")}</div>
        </div>
        """
        for label, value in fields
    )
    st.markdown(
        f"""
        <div class="studio-card">
            <div class="studio-eyebrow">{escape(eyebrow)}</div>
            <div class="studio-title" dir="auto">{escape(title)}</div>
            <div class="studio-kv">{items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_row(label: str, items: list[str], muted: bool = False) -> None:
    cleaned_items = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned_items:
        return

    chip_class = "studio-chip studio-chip--muted" if muted else "studio-chip"
    chips = "".join(
        f'<span class="{chip_class}">{escape(item)}</span>'
        for item in cleaned_items
    )
    st.markdown(
        f"""
        <div class="studio-card" style="padding: 14px 16px;">
            <div class="studio-eyebrow">{escape(label)}</div>
            <div class="studio-chip-wrap">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str | None = None, soft: bool = False) -> None:
    body_markup = (
        f'<div class="studio-panel-copy">{escape(body)}</div>'
        if body and str(body).strip()
        else ""
    )
    classes = "studio-panel studio-panel--soft" if soft else "studio-panel"
    st.markdown(
        f"""
        <div class="{classes}">
            <div class="studio-panel-title">{escape(title)}</div>
            {body_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_draft_option(draft: dict) -> str:
    title = str(draft.get("metadata", {}).get("title", "(untitled)")).strip() or "(untitled)"
    created_at = str(draft.get("created_at", "")).replace("T", " ")
    basename = os.path.basename(str(draft.get("video_path", "")))
    return f"{created_at} | {title} | {basename}"


def sync_scene_rows_from_script(script_text: str, existing_scenes: list[dict] | None = None) -> list[dict]:
    units = [
        part.strip(" \t\r\n-")
        for part in re.split(r"(?<=[\.\!\?\u061f])\s+|\n+", str(script_text or "").strip())
        if part and part.strip()
    ]
    if not units:
        units = [""] if str(script_text or "").strip() else []

    existing = list(existing_scenes or [])
    rows = []
    for index, unit in enumerate(units):
        previous = existing[index] if index < len(existing) and isinstance(existing[index], dict) else {}
        rows.append(
            {
                "scene_text": unit,
                "image_prompt": str(previous.get("image_prompt", "") or unit).strip(),
                "asset_path": str(previous.get("asset_path", "") or ""),
                "asset_type": str(previous.get("asset_type", "") or ""),
                "asset_source": str(previous.get("asset_source", "") or ""),
                "pixabay_query": str(previous.get("pixabay_query", "") or ""),
                "pixabay_query_type": str(previous.get("pixabay_query_type", "") or ""),
                "scene_index": index,
            }
        )
    return rows


def sync_script_from_scene_rows(scene_rows: list[dict]) -> str:
    return "\n\n".join(
        str(scene.get("scene_text", "") or "").strip()
        for scene in list(scene_rows or [])
        if str(scene.get("scene_text", "") or "").strip()
    )


def build_youtube_editor_signature(payload: dict) -> str:
    signature_payload = {
        "workspace_dir": payload.get("workspace_dir", ""),
        "video_path": payload.get("video_path", ""),
        "updated_at": payload.get("updated_at", ""),
        "subject": payload.get("subject", ""),
        "script": payload.get("script", ""),
        "scene_count": len(payload.get("scenes", []) or []),
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def ensure_youtube_editor_state(editor_key: str, source_payload: dict) -> dict:
    normalized = build_youtube_editor_payload(source_payload)
    signature = build_youtube_editor_signature(normalized)
    current = st.session_state.get(editor_key)
    if not isinstance(current, dict) or current.get("_source_signature") != signature:
        st.session_state[editor_key] = {
            "_source_signature": signature,
            "workspace_dir": normalized.get("workspace_dir", ""),
            "video_path": normalized.get("video_path", ""),
            "tts_path": normalized.get("tts_path", ""),
            "subject": normalized.get("subject", ""),
            "script": normalized.get("script", ""),
            "metadata": dict(normalized.get("metadata", {}) or {}),
            "scenes": [
                {
                    "scene_text": str(scene.get("scene_text", "") or ""),
                    "image_prompt": str(scene.get("image_prompt", "") or ""),
                    "asset_path": str(scene.get("asset_path", "") or ""),
                    "asset_type": str(scene.get("asset_type", "") or ""),
                    "asset_source": str(scene.get("asset_source", "") or ""),
                    "pixabay_query": str(scene.get("pixabay_query", "") or ""),
                    "pixabay_query_type": str(scene.get("pixabay_query_type", "") or ""),
                    "scene_index": int(scene.get("scene_index", index) or index),
                }
                for index, scene in enumerate(normalized.get("scenes", []) or [])
                if isinstance(scene, dict)
            ],
        }
    return st.session_state[editor_key]


def editor_state_to_payload(editor_state: dict) -> dict:
    scenes = [
        scene for scene in list(editor_state.get("scenes", []) or [])
        if isinstance(scene, dict)
    ]
    scene_units = [
        str(scene.get("scene_text", "") or "").strip()
        for scene in scenes
        if str(scene.get("scene_text", "") or "").strip()
    ]
    image_prompts = [
        str(scene.get("image_prompt", "") or "").strip()
        for scene in scenes
        if str(scene.get("scene_text", "") or "").strip() or str(scene.get("image_prompt", "") or "").strip()
    ]
    if len(image_prompts) < len(scene_units):
        fallback_prompt = image_prompts[-1] if image_prompts else (scene_units[-1] if scene_units else "")
        image_prompts.extend([fallback_prompt] * (len(scene_units) - len(image_prompts)))
    return {
        "workspace_dir": str(editor_state.get("workspace_dir", "") or "").strip(),
        "video_path": str(editor_state.get("video_path", "") or "").strip(),
        "tts_path": str(editor_state.get("tts_path", "") or "").strip(),
        "subject": str(editor_state.get("subject", "") or "").strip(),
        "script": str(editor_state.get("script", "") or "").strip(),
        "metadata": dict(editor_state.get("metadata", {}) or {}),
        "scene_units": scene_units,
        "image_prompts": image_prompts[:len(scene_units)],
        "scenes": scenes,
    }


def persist_youtube_editor_result(account_id: str, youtube: YouTube) -> None:
    preview_state = {
        "account_id": account_id,
        "subject": getattr(youtube, "subject", ""),
        "script": getattr(youtube, "script", ""),
        "metadata": youtube.metadata,
        "scene_units": list(getattr(youtube, "scene_units", []) or []),
        "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
        "workspace_dir": getattr(youtube, "_workspace_dir", ""),
        "video_path": getattr(youtube, "video_path", ""),
    }
    st.session_state["youtube_preview"] = preview_state

    if getattr(youtube, "video_path", ""):
        st.session_state["youtube_generated_video"] = {
            **preview_state,
            "video_path": youtube.video_path,
        }
        save_youtube_draft(
            account_id,
            youtube.video_path,
            youtube.metadata,
            workspace_dir=getattr(youtube, "_workspace_dir", ""),
            subject=getattr(youtube, "subject", ""),
            script=getattr(youtube, "script", ""),
            scene_units=list(getattr(youtube, "scene_units", []) or []),
            image_prompts=list(getattr(youtube, "image_prompts", []) or []),
        )


def render_youtube_scene_editor(
    *,
    editor_key: str,
    source_payload: dict,
    title: str,
    body: str,
    show_video: bool = True,
    allow_generate_from_editor: bool = False,
) -> tuple[dict, dict | None]:
    editor_state = ensure_youtube_editor_state(editor_key, source_payload)

    video_path = str(editor_state.get("video_path", "") or "").strip()
    workspace_dir = str(editor_state.get("workspace_dir", "") or "").strip()
    has_workspace = bool(workspace_dir and os.path.exists(os.path.join(workspace_dir, "run_state.json")))

    # ── Top action bar ──────────────────────────────────────────────────
    if has_workspace:
        btn_cols = st.columns([0.8, 1.2, 1.2, 1])
        action_save = btn_cols[0].button("Save", width="stretch", key=f"{editor_key}_save", type="primary")
        action_regen_voice = btn_cols[1].button("Regen Voice + Video", width="stretch", key=f"{editor_key}_voice")
        action_regen_scenes = btn_cols[2].button("Regen Scenes + Video", width="stretch", key=f"{editor_key}_scenes")
        if allow_generate_from_editor:
            action_generate = btn_cols[3].button("Generate Full Video", width="stretch", key=f"{editor_key}_generate")
        else:
            action_generate = False
    else:
        btn_cols = st.columns([1, 1])
        action_save = btn_cols[0].button("Save", width="stretch", key=f"{editor_key}_save", type="primary")
        if allow_generate_from_editor:
            action_generate = btn_cols[1].button("Generate Full Video", width="stretch", key=f"{editor_key}_generate")
        else:
            action_generate = False
        action_regen_voice = False
        action_regen_scenes = False

    # ── Video preview + metadata side by side ───────────────────────────
    video_col, meta_col = st.columns([0.38, 0.62])
    with video_col:
        if show_video and video_path and os.path.exists(video_path):
            st.video(video_path)
        elif show_video and workspace_dir:
            st.caption(f"Workspace: {workspace_dir}")

    with meta_col:
        subject_value = st.text_input(
            "Topic",
            value=editor_state.get("subject", ""),
            key=f"{editor_key}_subject",
        )
        metadata = dict(editor_state.get("metadata", {}) or {})
        title_col, desc_col = st.columns([1, 1])
        with title_col:
            metadata["title"] = st.text_input(
                "Title",
                value=metadata.get("title", ""),
                key=f"{editor_key}_title",
            )
        with desc_col:
            metadata["description"] = st.text_area(
                "Description",
                value=metadata.get("description", ""),
                height=80,
                key=f"{editor_key}_description",
            )
        script_value = st.text_area(
            "Transcript",
            value=editor_state.get("script", ""),
            height=160,
            key=f"{editor_key}_script",
        )

    editor_state["subject"] = subject_value
    editor_state["script"] = script_value
    editor_state["metadata"] = metadata

    # ── Scene sync actions ──────────────────────────────────────────────
    scene_action_cols = st.columns([1, 1, 0.7])
    if scene_action_cols[0].button("Sync Scenes From Transcript", width="stretch", key=f"{editor_key}_sync_scenes"):
        editor_state["scenes"] = sync_scene_rows_from_script(editor_state.get("script", ""), editor_state.get("scenes", []))
        st.session_state[editor_key] = editor_state
        st.rerun()
    if scene_action_cols[1].button("Sync Transcript From Scenes", width="stretch", key=f"{editor_key}_sync_script"):
        editor_state["script"] = sync_script_from_scene_rows(editor_state.get("scenes", []))
        st.session_state[editor_key] = editor_state
        st.rerun()
    if scene_action_cols[2].button("Add Scene", width="stretch", key=f"{editor_key}_add_scene"):
        next_index = len(editor_state.get("scenes", []))
        editor_state.setdefault("scenes", []).append(
            {
                "scene_text": "",
                "image_prompt": "",
                "asset_path": "",
                "asset_type": "",
                "asset_source": "",
                "scene_index": next_index,
            }
        )
        st.session_state[editor_key] = editor_state
        st.rerun()

    # ── Scene navigator (single scene at a time) ───────────────────────
    scenes = list(editor_state.get("scenes", []) or [])
    if not scenes:
        st.info("No scene rows yet. Click **Sync Scenes From Transcript** to create them.")
    else:
        scene_nav_key = f"{editor_key}_scene_nav"
        current_scene_idx = st.session_state.get(scene_nav_key, 0)
        if current_scene_idx >= len(scenes):
            current_scene_idx = 0

        # Navigator: prev / selector / next
        nav_prev, nav_select, nav_next = st.columns([0.3, 1.4, 0.3])
        with nav_prev:
            if st.button("< Prev", width="stretch", key=f"{editor_key}_prev", disabled=current_scene_idx == 0):
                st.session_state[scene_nav_key] = max(0, current_scene_idx - 1)
                st.rerun()
        with nav_select:
            scene_options = [
                f"Scene {i + 1}/{len(scenes)} — {(s.get('asset_source') or 'No asset')} {(s.get('asset_type') or '')}"
                for i, s in enumerate(scenes)
            ]
            selected_scene_label = st.selectbox(
                "Scene",
                options=scene_options,
                index=current_scene_idx,
                key=f"{editor_key}_scene_select",
                label_visibility="collapsed",
            )
            new_idx = scene_options.index(selected_scene_label) if selected_scene_label in scene_options else current_scene_idx
            if new_idx != current_scene_idx:
                st.session_state[scene_nav_key] = new_idx
                st.rerun()
            current_scene_idx = new_idx
        with nav_next:
            if st.button("Next >", width="stretch", key=f"{editor_key}_next", disabled=current_scene_idx >= len(scenes) - 1):
                st.session_state[scene_nav_key] = min(len(scenes) - 1, current_scene_idx + 1)
                st.rerun()

        # ── Current scene editor ────────────────────────────────────────
        index = current_scene_idx
        scene = scenes[index]
        asset_path = str(scene.get("asset_path", "") or "").strip()
        asset_ext = os.path.splitext(asset_path)[1].lower()
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        video_extensions = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}

        scene_media_col, scene_edit_col = st.columns([0.38, 0.62])
        with scene_media_col:
            if asset_path and os.path.exists(asset_path):
                if asset_ext in image_extensions:
                    st.image(asset_path, use_container_width=True)
                elif asset_ext in video_extensions:
                    st.video(asset_path)
                source_label = scene.get("asset_source", "Saved")
                type_label = scene.get("asset_type", "asset")
                st.caption(f"{source_label} {type_label}")
            elif asset_path:
                st.caption(f"Asset not found: {os.path.basename(asset_path)}")
            else:
                st.markdown(
                    '<div style="padding:2rem 1rem;text-align:center;color:#9ab0a4;border:2px dashed #cde0d5;border-radius:16px;">'
                    'No asset</div>',
                    unsafe_allow_html=True,
                )

        with scene_edit_col:
            transcript_value = st.text_area(
                "Scene transcript",
                value=str(scene.get("scene_text", "") or ""),
                height=100,
                key=f"{editor_key}_scene_text_{index}",
            )
            prompt_value = st.text_area(
                "Visual prompt",
                value=str(scene.get("image_prompt", "") or ""),
                height=90,
                key=f"{editor_key}_scene_prompt_{index}",
            )
            scene["scene_text"] = transcript_value
            scene["image_prompt"] = prompt_value
            scene["scene_index"] = index
            editor_state["scenes"][index] = scene

            # Scene actions row
            regen_col, dup_col, rm_col, clear_col = st.columns(4)
            if has_workspace:
                regen_provider = regen_col.selectbox(
                    "Regen with",
                    options=["Pixabay", "Nano Banana"],
                    index=0 if str(scene.get("asset_source", "")).strip().lower() in ("pixabay", "pexels") else 1,
                    key=f"{editor_key}_regen_provider_{index}",
                    label_visibility="collapsed",
                )
            if dup_col.button("Duplicate", width="stretch", key=f"{editor_key}_duplicate_{index}"):
                duplicated = dict(scene)
                duplicated["asset_path"] = ""
                duplicated["asset_source"] = ""
                duplicated["asset_type"] = ""
                editor_state["scenes"].insert(index + 1, duplicated)
                st.session_state[editor_key] = editor_state
                st.rerun()
            if rm_col.button("Remove", width="stretch", key=f"{editor_key}_remove_{index}") and len(scenes) > 1:
                editor_state["scenes"].pop(index)
                st.session_state[editor_key] = editor_state
                st.session_state[scene_nav_key] = min(index, len(editor_state["scenes"]) - 1)
                st.rerun()
            if clear_col.button("Clear Asset", width="stretch", key=f"{editor_key}_clear_asset_{index}"):
                editor_state["scenes"][index]["asset_path"] = ""
                editor_state["scenes"][index]["asset_source"] = ""
                editor_state["scenes"][index]["asset_type"] = ""
                st.session_state[editor_key] = editor_state
                st.rerun()

            if has_workspace:
                if regen_col.button("Regen Asset", width="stretch", key=f"{editor_key}_regen_asset_{index}"):
                    st.session_state[editor_key] = editor_state
                    provider_value = "pixabay" if regen_provider == "Pixabay" else "nanobanana2"
                    return editor_state_to_payload(editor_state), {
                        "type": "regenerate_scene_asset",
                        "scene_index": index,
                        "provider": provider_value,
                    }

    # ── Resolve action ──────────────────────────────────────────────────
    st.session_state[editor_key] = editor_state
    payload = editor_state_to_payload(editor_state)

    action: dict | None = None
    if action_save:
        action = {"type": "save"}
    elif action_generate:
        action = {"type": "generate_from_editor"}
    elif action_regen_voice:
        action = {"type": "regenerate_voiceover"}
    elif action_regen_scenes:
        action = {"type": "regenerate_scenes"}

    return payload, action


@st.dialog("Choose Pixabay Asset")
def render_youtube_pixabay_picker_dialog(
    account: dict,
    niche_override: str,
    language_override: str,
    dialect_override: str,
    context_override: str,
) -> None:
    picker_state = st.session_state.get(YOUTUBE_PIXABAY_PICKER_STATE_KEY, {})
    if not isinstance(picker_state, dict) or picker_state.get("account_id") != account["id"]:
        return

    scene_index = int(picker_state.get("scene_index", 0) or 0)
    candidates = list(picker_state.get("candidates", []) or [])
    editor_payload = dict(picker_state.get("editor_payload", {}) or {})
    editor_key = str(picker_state.get("editor_key", "") or "")
    query_source = str(picker_state.get("pixabay_query_source", "heuristic") or "heuristic").strip()
    search_mode = str(picker_state.get("search_mode", "primary") or "primary").strip()
    query_generation_note = str(picker_state.get("query_generation_note", "") or "").strip()
    query_variants = list(picker_state.get("query_variants", []) or [])

    st.caption(f"Scene {scene_index + 1}")
    st.write(str(picker_state.get("scene_text", "") or ""))
    st.text_area(
        "Prompt",
        value=str(picker_state.get("prompt_text", "") or ""),
        height=90,
        disabled=True,
        key=f"pixabay_picker_prompt_{account['id']}_{scene_index}",
    )
    st.caption(f"Query source: {query_source} | Search mode: {search_mode}")
    if query_generation_note:
        st.caption(query_generation_note)
    if query_variants:
        st.caption(
            "Tried q values: "
            + " | ".join(
                str(variant.get("query", "") or "").strip()
                for variant in query_variants[:5]
                if isinstance(variant, dict) and str(variant.get("query", "") or "").strip()
            )
        )

    if not candidates:
        st.info("No Pixabay candidates passed the quality threshold for this scene.")
        if st.button("Close", width="stretch", key=f"pixabay_picker_close_{account['id']}_{scene_index}"):
            st.session_state.pop(YOUTUBE_PIXABAY_PICKER_STATE_KEY, None)
            st.rerun()
    else:
        selected_index = st.radio(
            "Pick one fetched Pixabay asset",
            options=list(range(len(candidates))),
            format_func=lambda idx: (
                f"Option {idx + 1} | {candidates[idx].get('source', 'Pixabay')} "
                f"{candidates[idx].get('asset_type', 'asset')} | "
                f"score {float(candidates[idx].get('selection_score', 0.0) or 0.0):.1f}"
            ),
            key=f"pixabay_picker_choice_{account['id']}_{scene_index}",
        )
        selected_candidate = candidates[selected_index]
        preview_url = str(selected_candidate.get("preview_url", "") or "")
        if preview_url:
            if str(selected_candidate.get("asset_type", "")).lower() == "video":
                media_col, _ = st.columns([0.35, 0.65])
                media_col.video(preview_url)
            else:
                media_col, _ = st.columns([0.35, 0.65])
                media_col.image(preview_url, width="stretch")
        st.caption(
            f"Query: {selected_candidate.get('best_query', '')} | "
            f"Tags: {', '.join(selected_candidate.get('top_tags', []))}"
        )

        apply_col, cancel_col = st.columns([1, 1])
        if apply_col.button("Use This Asset", width="stretch", key=f"pixabay_picker_apply_{account['id']}_{scene_index}"):
            try:
                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=False,
                )
                with st.spinner("Applying selected Pixabay asset..."):
                    youtube.apply_workspace_edits(
                        editor_payload.get("workspace_dir", ""),
                        subject=editor_payload.get("subject", ""),
                        script=editor_payload.get("script", ""),
                        metadata=editor_payload.get("metadata", {}),
                        scene_units=editor_payload.get("scene_units", []),
                        image_prompts=editor_payload.get("image_prompts", []),
                    )
                    youtube.replace_scene_asset_with_pixabay_candidate(
                        scene_index,
                        selected_candidate,
                        editor_payload.get("workspace_dir", ""),
                    )
                persist_youtube_editor_result(account["id"], youtube)
                if editor_key:
                    st.session_state.pop(editor_key, None)
                st.session_state.pop(YOUTUBE_PIXABAY_PICKER_STATE_KEY, None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
                with st.expander("Technical details", expanded=False):
                    st.code(traceback.format_exc())

        if cancel_col.button("Cancel", width="stretch", key=f"pixabay_picker_cancel_{account['id']}_{scene_index}"):
            st.session_state.pop(YOUTUBE_PIXABAY_PICKER_STATE_KEY, None)
            st.rerun()


def load_pricing_for_video(video_path: str, metadata: dict | None = None) -> dict:
    if isinstance(metadata, dict) and isinstance(metadata.get("pricing"), dict):
        return metadata["pricing"]

    normalized_path = os.path.abspath(str(video_path or "").strip())
    if not normalized_path:
        return {}

    pricing_path = os.path.join(os.path.dirname(normalized_path), "pricing.json")
    if not os.path.exists(pricing_path):
        return {}

    try:
        with open(pricing_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def render_pricing_report(pricing: dict, key_prefix: str = "pricing") -> None:
    if not pricing:
        st.info("No pricing data is attached to this video yet.")
        return

    currency = str(pricing.get("currency", "USD") or "USD")
    total_cost = float(pricing.get("total_estimated_cost", 0.0) or 0.0)
    items = pricing.get("items", []) if isinstance(pricing.get("items"), list) else []
    summary = pricing.get("summary", {}) if isinstance(pricing.get("summary"), dict) else {}
    notes = pricing.get("notes", []) if isinstance(pricing.get("notes"), list) else []

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Estimated Total", f"{total_cost:.4f} {currency}")
    metric2.metric("Tracked Entries", len(items))
    metric3.metric("Billable Categories", len([item for item in summary.values() if float(item.get("estimated_cost", 0.0) or 0.0) > 0]))

    if summary:
        summary_rows = []
        for category, values in summary.items():
            summary_rows.append(
                {
                    "Category": category.replace("_", " ").title(),
                    "Estimated Cost": f"{float(values.get('estimated_cost', 0.0) or 0.0):.4f} {currency}",
                    "Entries": int(values.get("entries", 0) or 0),
                    "Quantity": float(values.get("quantity", 0.0) or 0.0),
                    "Units": ", ".join(values.get("units", [])),
                }
            )
        st.markdown("#### Category Summary")
        st.dataframe(summary_rows, width="stretch", hide_index=True, key=f"{key_prefix}_summary")

    if items:
        item_rows = []
        for item in items:
            item_rows.append(
                {
                    "Category": str(item.get("category", "")).replace("_", " ").title(),
                    "Type": item.get("type", ""),
                    "Provider": item.get("provider", ""),
                    "Model": item.get("model", ""),
                    "Quantity": item.get("quantity", 0),
                    "Unit": item.get("unit", ""),
                    "Estimated Cost": f"{float(item.get('estimated_cost', 0.0) or 0.0):.4f} {currency}",
                }
            )
        st.markdown("#### Line Items")
        st.dataframe(item_rows, width="stretch", hide_index=True, key=f"{key_prefix}_items")

    if notes:
        st.markdown("#### Notes")
        for note in notes:
            st.markdown(
                f"""
                <div class="studio-note" style="margin-bottom: 0.6rem;">
                    {escape(str(note))}
                </div>
                """,
                unsafe_allow_html=True,
            )


def ensure_llm_selected() -> tuple[str, str]:
    provider = get_llm_provider()
    model = get_configured_llm_model()

    if provider == "openai":
        model = model or "gpt-5-mini"
        select_model(model, provider=provider)
        return provider, model

    if not model:
        models = list_models(provider=provider)
        if not models:
            raise RuntimeError(
                "No Ollama models are available. Set ollama_model in config.json or pull a model first."
            )
        model = models[0]

    select_model(model, provider=provider)
    return provider, model


def get_youtube_metadata_model_options(config: dict) -> tuple[list[str], str, str | None]:
    provider = str(config.get("llm_provider", "ollama") or "ollama").strip().lower()
    if provider not in ("openai", "ollama"):
        provider = "ollama"

    active_model = str(
        config.get("openai_model", "") if provider == "openai" else config.get("ollama_model", "")
    ).strip()
    configured_model = str(config.get("youtube_metadata_model", "") or "").strip()

    discovered_models: list[str] = []
    warning_message = None
    try:
        discovered_models = list_models(provider=provider)
    except Exception as exc:
        warning_message = (
            f"Couldn't load live {provider.title()} models right now, so the metadata list is using saved values only: {exc}"
        )

    option_candidates: list[str] = []
    if discovered_models:
        if configured_model and configured_model in discovered_models:
            option_candidates.append(configured_model)
        if active_model:
            option_candidates.append(active_model)
        option_candidates.extend(discovered_models)
    else:
        if configured_model:
            option_candidates.append(configured_model)
        if active_model:
            option_candidates.append(active_model)

    options = [""]
    seen = {""}
    for model_name in option_candidates:
        if model_name and model_name not in seen:
            options.append(model_name)
            seen.add(model_name)

    if provider == "openai":
        auto_label = "Auto (gpt-5-nano)"
    elif active_model:
        auto_label = f"Auto ({active_model})"
    else:
        auto_label = "Auto (use the active Ollama model)"
    return options, auto_label, warning_message


def render_overview() -> None:
    config = load_config()
    provider = config.get("llm_provider", "ollama")
    active_model = config.get("openai_model") if provider == "openai" else config.get("ollama_model")
    render_hero(
        "MoneyPrinterPro Studio",
        "Manage accounts, generate short-form content, review drafts, and publish across Twitter, YouTube, and TikTok from one calmer workspace.",
        chips=[
            f"LLM: {provider}",
            f"Model: {active_model or 'Not set'}",
            f"Image provider: {config.get('image_provider', 'nanobanana2')}",
            f"TTS: {config.get('tts_provider', 'auto')}",
        ],
        eyebrow="Control Room",
    )

    twitter_accounts = get_accounts("twitter")
    youtube_accounts = get_accounts("youtube")
    tiktok_accounts = get_accounts("tiktok")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Twitter Accounts", len(twitter_accounts))
    col2.metric("YouTube Accounts", len(youtube_accounts))
    col3.metric("TikTok Accounts", len(tiktok_accounts))
    col4.metric("LLM Provider", provider)
    col5.metric("Active Model", active_model or "Not set")

    note_col, guide_col = st.columns([1.2, 1])
    with note_col:
        st.markdown(
            """
            <div class="studio-note">
                Close Firefox before using browser automation actions like <strong>Post</strong> or <strong>Upload</strong>.
                Preview, generation, and metadata work can all be done without launching a browser first.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_col:
        render_kv_card(
            "Quick Start",
            "Best flow for smooth runs",
            [
                ("1", "Set providers and defaults in Config"),
                ("2", "Create or update your channel/account profiles"),
                ("3", "Generate previews before posting or uploading"),
                ("4", "Keep Firefox closed until the final browser action"),
            ],
        )


def render_config_tab() -> None:
    render_section_intro(
        "Configuration",
        "Set the core providers and defaults here. Keep expensive providers only where quality matters, and use the lighter options for metadata or retries.",
        eyebrow="Settings",
    )
    config = load_config()
    pricing_defaults = get_pricing_config()
    if not isinstance(config.get("pricing"), dict):
        config["pricing"] = pricing_defaults

    def as_int(value: object, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def as_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    current_target_duration = max(0, as_int(config.get("youtube_target_duration_seconds", 30), 30))
    current_min_prompts = max(1, as_int(config.get("min_image_prompts", 10), 10))
    current_max_prompts = max(current_min_prompts, as_int(config.get("max_image_prompts", 12), 12))

    top_summary_col, top_note_col = st.columns([1.15, 0.85])
    with top_summary_col:
        render_kv_card(
            "Current Defaults",
            "Global automation profile",
            [
                ("LLM provider", str(config.get("llm_provider", "ollama"))),
                ("Image provider", str(config.get("image_provider", "nanobanana2"))),
                ("Asset strategy", str(config.get("asset_strategy", "mixed"))),
                ("TTS provider", str(config.get("tts_provider", "auto"))),
                ("Target duration", f"{config.get('youtube_target_duration_seconds', 30)}s"),
            ],
        )
    with top_note_col:
        st.markdown(
            """
            <div class="studio-note">
                Keep this page for defaults and provider setup only. Channel voice, dialect, and publishing choices are usually better handled in the account and studio tabs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("config_form"):
        core_tab, providers_tab, video_tab, pricing_tab, advanced_tab = st.tabs(
            ["Core Setup", "Providers + Keys", "Voice + Video", "Pricing", "Advanced"]
        )

        with core_tab:
            left, right = st.columns(2)
            with left:
                render_panel("Browser + Core", "The defaults used when an account does not override them.", soft=True)
                config["firefox_profile"] = st.text_input(
                    "Default Firefox profile",
                    value=config.get("firefox_profile", ""),
                    help="Fallback profile path used when a specific account does not define one.",
                )
                config["headless"] = st.checkbox("Headless browser", value=config.get("headless", False))
                config["verbose"] = st.checkbox("Verbose logs", value=config.get("verbose", True))
                config["llm_provider"] = st.selectbox(
                    "LLM provider",
                    options=["openai", "ollama"],
                    index=0 if config.get("llm_provider", "ollama") == "openai" else 1,
                )
                config["openai_model"] = st.text_input(
                    "OpenAI model",
                    value=config.get("openai_model", "gpt-5-mini"),
                )
                config["ollama_model"] = st.text_input(
                    "Ollama model",
                    value=config.get("ollama_model", ""),
                )
            with right:
                render_panel("Language Defaults", "These values mainly shape social content defaults and narration guidance.", soft=True)
                config["twitter_language"] = st.text_input(
                    "Twitter language",
                    value=config.get("twitter_language", "English"),
                )
                config["twitter_dialect"] = st.text_input(
                    "Twitter dialect",
                    value=config.get("twitter_dialect", ""),
                )
                metadata_model_options, metadata_auto_label, metadata_model_warning = get_youtube_metadata_model_options(config)
                current_metadata_model = str(config.get("youtube_metadata_model", "") or "").strip()
                metadata_model_index = (
                    metadata_model_options.index(current_metadata_model)
                    if current_metadata_model in metadata_model_options
                    else 0
                )
                config["youtube_metadata_model"] = st.selectbox(
                    "YouTube metadata model",
                    options=metadata_model_options,
                    index=metadata_model_index,
                    format_func=lambda model_name: metadata_auto_label if model_name == "" else model_name,
                    help="Optional lighter model override for YouTube title/description/tags. Choices follow the selected LLM provider.",
                )
                st.caption(
                    "This list follows the selected LLM provider. If you change the provider or base model above, refresh the metadata choices before saving."
                )
                if metadata_model_warning:
                    st.caption(metadata_model_warning)

        with providers_tab:
            left, right = st.columns(2)
            with left:
                render_panel("Image Providers", "Choose the main image path and the stock-vs-AI strategy.", soft=True)
                config["image_provider"] = st.selectbox(
                    "Image provider",
                    options=["openai", "openrouter", "nanobanana2"],
                    index=["openai", "openrouter", "nanobanana2"].index(
                        config.get("image_provider", "nanobanana2")
                        if config.get("image_provider", "nanobanana2") in ["openai", "openrouter", "nanobanana2"]
                        else "nanobanana2"
                    ),
                )
                config["asset_strategy"] = st.selectbox(
                    "Visual asset strategy",
                    options=["mixed", "pixabay_only", "ai_only"],
                    index=["mixed", "pixabay_only", "ai_only"].index(
                        config.get("asset_strategy", "mixed")
                        if config.get("asset_strategy", "mixed") in ["mixed", "pixabay_only", "ai_only"]
                        else "mixed"
                    ),
                    help="Use free Pixabay stock first, AI only, or a mixed cost-saving approach.",
                )
                config["max_ai_assets"] = int(
                    st.number_input(
                        "Max AI visual assets",
                        min_value=0,
                        max_value=60,
                        step=1,
                        value=min(60, max(0, as_int(config.get("max_ai_assets", 2), 2))),
                        help="Used by the mixed strategy to limit how many AI-generated images are allowed per video.",
                    )
                )
                config["openai_image_model"] = st.text_input(
                    "OpenAI image model",
                    value=config.get("openai_image_model", "gpt-image-1"),
                )
                config["openrouter_image_model"] = st.text_input(
                    "OpenRouter image model",
                    value=config.get("openrouter_image_model", "black-forest-labs/flux.2-flex"),
                )
                config["nanobanana2_model"] = st.text_input(
                    "Nano Banana model",
                    value=config.get("nanobanana2_model", "gemini-3.1-flash-image-preview"),
                )
                config["openai_image_quality"] = st.selectbox(
                    "OpenAI image quality",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(
                        config.get("openai_image_quality", "low")
                        if config.get("openai_image_quality", "low") in ["low", "medium", "high"]
                        else "low"
                    ),
                )
            with right:
                render_panel("Keys + Endpoints", "Sensitive keys stay grouped here so the rest of the page stays lighter to scan.", soft=True)
                config["openai_base_url"] = st.text_input(
                    "OpenAI base URL",
                    value=config.get("openai_base_url", "https://api.openai.com/v1"),
                )
                config["ollama_base_url"] = st.text_input(
                    "Ollama base URL",
                    value=config.get("ollama_base_url", "http://127.0.0.1:11434"),
                )
                config["openai_api_key"] = st.text_input(
                    "OpenAI API key",
                    value=config.get("openai_api_key", ""),
                    type="password",
                )
                config["openrouter_api_key"] = st.text_input(
                    "OpenRouter API key",
                    value=config.get("openrouter_api_key", ""),
                    type="password",
                )
                config["pixabay_api_key"] = st.text_input(
                    "Pixabay API key",
                    value=config.get("pixabay_api_key", ""),
                    type="password",
                )
                config["youtube_data_api_key"] = st.text_input(
                    "YouTube Data API key (Trends)",
                    value=config.get("youtube_data_api_key", ""),
                    type="password",
                    help="Enables YouTube trending in the Trends tab. Create one free at console.cloud.google.com → enable 'YouTube Data API v3'. Reddit trends work without a key.",
                )
                config["trends_region"] = st.text_input(
                    "Trends region code",
                    value=config.get("trends_region", "US"),
                    help="ISO region for YouTube trends (e.g. US, EG, GB).",
                )
                config["reddit_client_id"] = st.text_input(
                    "Reddit client ID (Trends)",
                    value=config.get("reddit_client_id", ""),
                    type="password",
                    help="Reddit blocks unauthenticated access. Create a free app at reddit.com/prefs/apps (type 'script') and paste its client ID + secret here.",
                )
                config["reddit_client_secret"] = st.text_input(
                    "Reddit client secret (Trends)",
                    value=config.get("reddit_client_secret", ""),
                    type="password",
                )

        with video_tab:
            left, right = st.columns(2)
            with left:
                render_panel("Narration", "Set the speech provider and the subtitle look that will be burned into videos.", soft=True)
                config["tts_provider"] = st.selectbox(
                    "TTS provider",
                    options=["auto", "openai", "kitten"],
                    index=["auto", "openai", "kitten"].index(
                        config.get("tts_provider", "auto")
                        if config.get("tts_provider", "auto") in ["auto", "openai", "kitten"]
                        else "auto"
                    ),
                )
                config["openai_tts_model"] = st.text_input(
                    "OpenAI TTS model",
                    value=config.get("openai_tts_model", "gpt-4o-mini-tts"),
                )
                config["openai_tts_voice"] = st.text_input(
                    "OpenAI TTS voice",
                    value=config.get("openai_tts_voice", "onyx"),
                )
                stt_options = ["local_whisper", "script_based", "third_party_assemblyai"]
                current_stt = config.get("stt_provider", "local_whisper")
                if current_stt not in stt_options:
                    current_stt = "local_whisper"
                config["stt_provider"] = st.selectbox(
                    "Subtitle provider (STT)",
                    options=stt_options,
                    index=stt_options.index(current_stt),
                    format_func=lambda s: {"local_whisper": "Local Whisper (GPU, accurate sync)", "script_based": "Script-based (fast, less accurate)", "third_party_assemblyai": "AssemblyAI (cloud, paid)"}.get(s, s),
                    help="Local Whisper uses your GPU for accurate subtitle timing. Script-based estimates from text.",
                )
                subtitle_mode_options = ["word_by_word", "chunk"]
                current_subtitle_mode = config.get("subtitle_mode", "word_by_word")
                if current_subtitle_mode not in subtitle_mode_options:
                    current_subtitle_mode = "word_by_word"
                config["subtitle_mode"] = st.selectbox(
                    "Subtitle timing mode",
                    options=subtitle_mode_options,
                    index=subtitle_mode_options.index(current_subtitle_mode),
                    help="word_by_word shows one word at a time. chunk shows short grouped phrases.",
                )
                config["subtitle_font_english"] = st.text_input(
                    "English subtitle font file",
                    value=config.get("subtitle_font_english", config.get("subtitle_font", config.get("font", "bold_font.ttf"))),
                )
                config["subtitle_font_arabic"] = st.text_input(
                    "Arabic subtitle font file",
                    value=config.get("subtitle_font_arabic", config.get("subtitle_font", config.get("font", "bold_font.ttf"))),
                )
                config["subtitle_font_size"] = int(
                    st.number_input(
                        "Subtitle font size",
                        min_value=24,
                        max_value=180,
                        step=2,
                        value=max(24, as_int(config.get("subtitle_font_size", 84), 84)),
                    )
                )
                config["subtitle_color"] = st.text_input(
                    "Subtitle color",
                    value=config.get("subtitle_color", "#FFF7D6"),
                )
                config["subtitle_stroke_color"] = st.text_input(
                    "Subtitle stroke color",
                    value=config.get("subtitle_stroke_color", "#000000"),
                )
                config["subtitle_stroke_width"] = int(
                    st.number_input(
                        "Subtitle stroke width",
                        min_value=0,
                        max_value=20,
                        step=1,
                        value=max(0, as_int(config.get("subtitle_stroke_width", 6), 6)),
                    )
                )
            with right:
                render_panel("Video Rhythm", "These defaults mostly affect pacing, asset count, and search breadth.", soft=True)
                config["youtube_target_duration_seconds"] = int(
                    st.number_input(
                        "Target video duration (seconds)",
                        min_value=0,
                        max_value=300,
                        step=5,
                        value=current_target_duration,
                        help="Guides script generation toward a target spoken runtime. Set 0 to disable.",
                    )
                )
                config["min_image_prompts"] = int(
                    st.number_input(
                        "Min image prompts",
                        min_value=1,
                        max_value=30,
                        step=1,
                        value=current_min_prompts,
                    )
                )
                config["max_image_prompts"] = int(
                    st.number_input(
                        "Max image prompts",
                        min_value=config["min_image_prompts"],
                        max_value=40,
                        step=1,
                        value=max(config["min_image_prompts"], current_max_prompts),
                    )
                )
                config["pixabay_results_per_query"] = int(
                    st.number_input(
                        "Pixabay results per query",
                        min_value=3,
                        max_value=20,
                        step=1,
                        value=min(20, max(3, as_int(config.get("pixabay_results_per_query", 6), 6))),
                    )
                )
                config["subtitle_font"] = st.text_input(
                    "Legacy subtitle font file",
                    value=config.get("subtitle_font", config.get("font", "bold_font.ttf")),
                    help="Fallback font file kept for backward compatibility.",
                )

                st.markdown("---")
                render_panel("Video Quality", "Frame rate, transitions, and sound effects that shape the final output.", soft=True)
                config["video_fps"] = int(
                    st.selectbox(
                        "Video FPS",
                        options=[24, 30, 60],
                        index=[24, 30, 60].index(
                            config.get("video_fps", 30)
                            if config.get("video_fps", 30) in [24, 30, 60]
                            else 30
                        ),
                        help="30 is standard. 60 gives smoother playback, especially on TikTok.",
                    )
                )
                config["crossfade_duration"] = as_float(
                    st.number_input(
                        "Crossfade duration (seconds)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.05,
                        value=as_float(config.get("crossfade_duration", 0.3), 0.3),
                        help="Fade transition between scenes. 0 = hard cut.",
                    ),
                    0.3,
                )
                config["sound_effects_enabled"] = st.checkbox(
                    "Enable sound effects",
                    value=config.get("sound_effects_enabled", False),
                    help="Add sound effects triggered by specific words in the voiceover.",
                )
                config["sound_effects_volume"] = as_float(
                    st.number_input(
                        "Sound effects volume",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.05,
                        value=as_float(config.get("sound_effects_volume", 0.45), 0.45),
                        help="How loud sound effects are relative to the voiceover (0-1).",
                    ),
                    0.45,
                )
                config["sound_effects_offset"] = as_float(
                    st.number_input(
                        "Sound effects timing offset (seconds)",
                        min_value=-1.0,
                        max_value=1.0,
                        step=0.05,
                        value=as_float(config.get("sound_effects_offset", -0.15), -0.15),
                        help="Negative = SFX plays before the trigger word (anticipation). Positive = after.",
                    ),
                    -0.15,
                )

        with pricing_tab:
            pricing = config.setdefault("pricing", {})
            pricing.setdefault("currency", pricing_defaults.get("currency", "USD"))
            text_generation = pricing.setdefault("text_generation", {})
            openai_text = text_generation.setdefault(
                "openai",
                pricing_defaults.get("text_generation", {}).get("openai", {}),
            )
            ollama_text = text_generation.setdefault(
                "ollama",
                pricing_defaults.get("text_generation", {}).get("ollama", {}),
            )
            openai_text_models = openai_text.setdefault("models", {})
            ollama_text_models = ollama_text.setdefault("models", {})
            gpt5mini_pricing = openai_text_models.setdefault(
                "gpt-5-mini",
                pricing_defaults["text_generation"]["openai"]["models"]["gpt-5-mini"],
            )
            gpt5nano_pricing = openai_text_models.setdefault(
                "gpt-5-nano",
                pricing_defaults["text_generation"]["openai"]["models"]["gpt-5-nano"],
            )
            ollama_default_pricing = ollama_text_models.setdefault(
                "default",
                pricing_defaults["text_generation"]["ollama"]["models"]["default"],
            )

            image_generation = pricing.setdefault("image_generation", {})
            openai_images = image_generation.setdefault(
                "openai",
                pricing_defaults["image_generation"]["openai"],
            )
            nanobanana_images = image_generation.setdefault(
                "nanobanana2",
                pricing_defaults["image_generation"]["nanobanana2"],
            )
            openrouter_images = image_generation.setdefault(
                "openrouter",
                pricing_defaults["image_generation"]["openrouter"],
            )
            pixabay_images = image_generation.setdefault(
                "pixabay",
                pricing_defaults["image_generation"]["pixabay"],
            )
            openai_image_model_pricing = openai_images.setdefault("models", {}).setdefault(
                "gpt-image-1",
                pricing_defaults["image_generation"]["openai"]["models"]["gpt-image-1"],
            )
            openai_image_qualities = openai_image_model_pricing.setdefault(
                "qualities",
                pricing_defaults["image_generation"]["openai"]["models"]["gpt-image-1"]["qualities"],
            )
            nanobanana_default_pricing = nanobanana_images.setdefault("models", {}).setdefault(
                "gemini-3.1-flash-image-preview",
                pricing_defaults["image_generation"]["nanobanana2"]["models"]["gemini-3.1-flash-image-preview"],
            )
            openrouter_default_pricing = openrouter_images.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["image_generation"]["openrouter"]["models"]["default"],
            )
            pixabay_default_pricing = pixabay_images.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["image_generation"]["pixabay"]["models"]["default"],
            )

            tts_pricing = pricing.setdefault("tts", {})
            openai_tts = tts_pricing.setdefault("openai", pricing_defaults["tts"]["openai"])
            kitten_tts = tts_pricing.setdefault("kitten", pricing_defaults["tts"]["kitten"])
            openai_tts_default = openai_tts.setdefault("models", {}).setdefault(
                "gpt-4o-mini-tts",
                pricing_defaults["tts"]["openai"]["models"]["gpt-4o-mini-tts"],
            )
            kitten_tts_default = kitten_tts.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["tts"]["kitten"]["models"]["default"],
            )

            stt_pricing = pricing.setdefault("stt", {})
            script_based_stt = stt_pricing.setdefault(
                "script_based",
                pricing_defaults["stt"]["script_based"],
            )
            whisper_stt = stt_pricing.setdefault(
                "local_whisper",
                pricing_defaults["stt"]["local_whisper"],
            )
            assemblyai_stt = stt_pricing.setdefault(
                "third_party_assemblyai",
                pricing_defaults["stt"]["third_party_assemblyai"],
            )
            script_based_default = script_based_stt.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["stt"]["script_based"]["models"]["default"],
            )
            whisper_default = whisper_stt.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["stt"]["local_whisper"]["models"]["default"],
            )
            assemblyai_default = assemblyai_stt.setdefault("models", {}).setdefault(
                "default",
                pricing_defaults["stt"]["third_party_assemblyai"]["models"]["default"],
            )

            left, right = st.columns(2)
            with left:
                render_panel(
                    "Tracked Rates",
                    "These values power the per-video cost estimate. Adjust them if your provider pricing or plan differs.",
                    soft=True,
                )
                pricing["currency"] = st.text_input(
                    "Pricing currency",
                    value=str(pricing.get("currency", pricing_defaults.get("currency", "USD"))),
                )
                st.markdown("##### Text Generation")
                gpt5mini_pricing["input_per_1m_tokens"] = as_float(
                    st.text_input(
                        "OpenAI gpt-5-mini input / 1M tokens",
                        value=str(gpt5mini_pricing.get("input_per_1m_tokens", 0.25)),
                    ),
                    0.25,
                )
                gpt5mini_pricing["output_per_1m_tokens"] = as_float(
                    st.text_input(
                        "OpenAI gpt-5-mini output / 1M tokens",
                        value=str(gpt5mini_pricing.get("output_per_1m_tokens", 2.0)),
                    ),
                    2.0,
                )
                gpt5nano_pricing["input_per_1m_tokens"] = as_float(
                    st.text_input(
                        "OpenAI gpt-5-nano input / 1M tokens",
                        value=str(gpt5nano_pricing.get("input_per_1m_tokens", 0.05)),
                    ),
                    0.05,
                )
                gpt5nano_pricing["output_per_1m_tokens"] = as_float(
                    st.text_input(
                        "OpenAI gpt-5-nano output / 1M tokens",
                        value=str(gpt5nano_pricing.get("output_per_1m_tokens", 0.4)),
                    ),
                    0.4,
                )
                ollama_default_pricing["input_per_1m_tokens"] = as_float(
                    st.text_input(
                        "Ollama default input / 1M tokens",
                        value=str(ollama_default_pricing.get("input_per_1m_tokens", 0.0)),
                        help="Leave at 0 if your Ollama usage is effectively flat-cost or you do not want to estimate it.",
                    ),
                    0.0,
                )
                ollama_default_pricing["output_per_1m_tokens"] = as_float(
                    st.text_input(
                        "Ollama default output / 1M tokens",
                        value=str(ollama_default_pricing.get("output_per_1m_tokens", 0.0)),
                    ),
                    0.0,
                )

                st.markdown("##### Speech + Subtitles")
                openai_tts_default["per_minute_audio"] = as_float(
                    st.text_input(
                        "OpenAI TTS cost / minute audio",
                        value=str(openai_tts_default.get("per_minute_audio", 0.015)),
                    ),
                    0.015,
                )
                kitten_tts_default["per_minute_audio"] = as_float(
                    st.text_input(
                        "Kitten TTS cost / minute audio",
                        value=str(kitten_tts_default.get("per_minute_audio", 0.0)),
                    ),
                    0.0,
                )
                script_based_default["per_minute_audio"] = as_float(
                    st.text_input(
                        "Script-based subtitles cost / minute audio",
                        value=str(script_based_default.get("per_minute_audio", 0.0)),
                    ),
                    0.0,
                )
                whisper_default["per_minute_audio"] = as_float(
                    st.text_input(
                        "Local Whisper cost / minute audio",
                        value=str(whisper_default.get("per_minute_audio", 0.0)),
                    ),
                    0.0,
                )
                assemblyai_default["per_minute_audio"] = as_float(
                    st.text_input(
                        "AssemblyAI cost / minute audio",
                        value=str(assemblyai_default.get("per_minute_audio", 0.0025)),
                    ),
                    0.0025,
                )

            with right:
                render_panel(
                    "Image Costs",
                    "AI image pricing is tracked per successful generated image. Pixabay defaults to free unless you want to assign an internal media cost.",
                    soft=True,
                )
                openai_image_qualities["low"] = as_float(
                    st.text_input(
                        "OpenAI image low quality / image",
                        value=str(openai_image_qualities.get("low", 0.016)),
                    ),
                    0.016,
                )
                openai_image_qualities["medium"] = as_float(
                    st.text_input(
                        "OpenAI image medium quality / image",
                        value=str(openai_image_qualities.get("medium", 0.063)),
                    ),
                    0.063,
                )
                openai_image_qualities["high"] = as_float(
                    st.text_input(
                        "OpenAI image high quality / image",
                        value=str(openai_image_qualities.get("high", 0.25)),
                    ),
                    0.25,
                )
                nanobanana_default_pricing["per_image"] = as_float(
                    st.text_input(
                        "Nano Banana cost / image",
                        value=str(nanobanana_default_pricing.get("per_image", 0.039)),
                    ),
                    0.039,
                )
                openrouter_default_pricing["per_image"] = as_float(
                    st.text_input(
                        "OpenRouter image cost / image",
                        value=str(openrouter_default_pricing.get("per_image", 0.0)),
                    ),
                    0.0,
                )
                pixabay_default_pricing["per_asset"] = as_float(
                    st.text_input(
                        "Pixabay stock asset cost / asset",
                        value=str(pixabay_default_pricing.get("per_asset", 0.0)),
                    ),
                    0.0,
                )
                st.markdown(
                    """
                    <div class="studio-note">
                        Tip: If you use an Ollama cloud model or any provider plan with custom pricing, set those numbers here so the report reflects your real spend more closely.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with advanced_tab:
            st.markdown(
                """
                <div class="studio-note">
                    Advanced values stay here so the main setup path stays short. Most day-to-day work should only need the first three groups.
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Raw config JSON", expanded=False):
                st.code(json.dumps(config, ensure_ascii=False, indent=2), language="json")

        refresh_col, save_col = st.columns(2)
        with refresh_col:
            refresh_metadata_choices = st.form_submit_button(
                "Refresh Metadata Choices",
                width="stretch",
            )
        with save_col:
            submitted = st.form_submit_button("Save Config", width="stretch")

    if refresh_metadata_choices:
        st.info("Metadata model choices were refreshed for the selected LLM provider and base model.")
    if submitted:
        save_config(config)
        st.success("Config saved.")


def render_account_editor(provider: str) -> None:
    provider_title = provider.title()
    render_section_intro(
        f"{provider_title} Accounts",
        "Keep each account’s voice, profile path, and defaults explicit so generation stays consistent over time.",
        eyebrow="Accounts",
    )
    accounts = get_accounts(provider)

    if accounts:
        selected_id = st.selectbox(
            f"Select {provider} account",
            options=[account["id"] for account in accounts],
            format_func=lambda account_id: next(
                f'{account["nickname"]} ({account["id"]})'
                for account in accounts
                if account["id"] == account_id
            ),
            key=f"{provider}_account_select",
        )
        selected_account = next(account for account in accounts if account["id"] == selected_id)

        summary_fields = [
            ("Nickname", str(selected_account.get("nickname", ""))),
            ("Profile", os.path.basename(str(selected_account.get("firefox_profile", ""))) or "Not set"),
        ]
        if provider == "twitter":
            summary_fields.extend(
                [
                    ("Topic", str(selected_account.get("topic", ""))),
                    ("Posts", str(len(selected_account.get("posts", [])))),
                ]
            )
        else:
            summary_fields.extend(
                [
                    ("Language", str(selected_account.get("language", "English"))),
                    ("Dialect", str(selected_account.get("dialect", "") or "Default")),
                    ("Videos", str(len(selected_account.get("videos", [])))),
                ]
            )
            if provider == "youtube":
                summary_fields.append(
                    ("Made for kids", "Yes" if selected_account.get("is_for_kids", get_is_for_kids()) else "No")
                )

        render_kv_card("Selected Account", selected_account.get("nickname", provider_title), summary_fields)

        with st.form(f"edit_{provider}_account"):
            st.markdown("##### Identity")
            nickname = st.text_input("Nickname", value=selected_account.get("nickname", ""))
            firefox_profile = st.text_input(
                "Firefox profile",
                value=selected_account.get("firefox_profile", ""),
            )
            character_context = st.text_area(
                "Character context",
                value=selected_account.get("character_context", ""),
                help="Describe the persona, tone, audience, recurring themes, and what this account should sound like.",
            )

            if provider == "twitter":
                st.markdown("##### Posting Defaults")
                topic = st.text_input("Topic", value=selected_account.get("topic", ""))
            else:
                st.markdown("##### Channel Defaults")
                niche = st.text_input("Niche", value=selected_account.get("niche", ""))
                language = st.text_input("Language", value=selected_account.get("language", "English"))
                dialect = st.text_input("Dialect", value=selected_account.get("dialect", ""))
                is_for_kids = (
                    st.checkbox("Made for kids by default", value=selected_account.get("is_for_kids", get_is_for_kids()))
                    if provider == "youtube"
                    else False
                )

            save_button = st.form_submit_button("Save Changes", width="stretch")

        if save_button:
            updates = {
                "nickname": nickname,
                "firefox_profile": firefox_profile,
                "character_context": character_context,
            }
            if provider == "twitter":
                updates["topic"] = topic
            else:
                updates["niche"] = niche
                updates["language"] = language
                updates["dialect"] = dialect
                if provider == "youtube":
                    updates["is_for_kids"] = is_for_kids

            update_account(provider, selected_id, updates)
            st.success("Account updated.")
            st.rerun()

        if st.button(
            f"Delete {provider.title()} Account",
            type="secondary",
            width="stretch",
            key=f"delete_{provider}_account",
        ):
            remove_account(provider, selected_id)
            st.warning("Account deleted.")
            st.rerun()

        with st.expander("Selected account JSON"):
            st.code(json.dumps(selected_account, ensure_ascii=False, indent=2), language="json")

    else:
        st.info(f"No {provider} accounts yet.")

    st.markdown("---")
    st.markdown(f"#### Create {provider_title} Account")

    with st.form(f"create_{provider}_account"):
        st.markdown("##### New account basics")
        nickname = st.text_input("Nickname", key=f"create_{provider}_nickname")
        firefox_profile = st.text_input("Firefox profile", key=f"create_{provider}_profile")
        character_context = st.text_area(
            "Character context",
            key=f"create_{provider}_character_context",
            help="Example: Smart, casual Egyptian creator who gives sharp opinions about trending TV drama and always sounds natural, witty, and current.",
        )

        if provider == "twitter":
            topic = st.text_input("Topic", key="create_twitter_topic")
        else:
            niche = st.text_input("Niche", key=f"create_{provider}_niche")
            language = st.text_input("Language", value="English", key=f"create_{provider}_language")
            dialect = st.text_input("Dialect", value="", key=f"create_{provider}_dialect")
            create_is_for_kids = (
                st.checkbox("Made for kids by default", value=False, key=f"create_{provider}_is_for_kids")
                if provider == "youtube"
                else False
            )

        create_button = st.form_submit_button(f"Create {provider.title()} Account", width="stretch")

    if create_button:
        account = {
            "id": str(uuid4()),
            "nickname": nickname,
            "firefox_profile": firefox_profile,
            "character_context": character_context,
        }
        if provider == "twitter":
            account["topic"] = topic
            account["posts"] = []
        else:
            account["niche"] = niche
            account["language"] = language
            account["dialect"] = dialect
            if provider == "youtube":
                account["is_for_kids"] = create_is_for_kids
            account["videos"] = []

        add_account(provider, account)
        st.success(f"{provider.title()} account created.")
        st.rerun()


def render_twitter_studio() -> None:
    accounts = get_accounts("twitter")
    if not accounts:
        st.info("Create a Twitter account first in the Accounts page.")
        return

    sel_col, stat_col = st.columns([3, 0.5])
    with sel_col:
        account_id = st.selectbox(
            "Twitter account",
            options=[account["id"] for account in accounts],
            format_func=lambda selected_id: next(
                f'{account["nickname"]} — {account.get("topic", "")}'
                for account in accounts
                if account["id"] == selected_id
            ),
        )
    account = next(account for account in accounts if account["id"] == account_id)
    with stat_col:
        st.metric("Posts", len(account.get("posts", [])))

    with st.expander("Overrides", expanded=False):
        topic_override = st.text_input("Topic", value=account.get("topic", ""))
        context_override = st.text_area(
            "Character context",
            value=account.get("character_context", ""),
            height=100,
        )

    render_kv_card(
        "Active Account",
        account.get("nickname", "Twitter account"),
        [
            ("Topic", (topic_override if topic_override != account.get("topic", "") else account.get("topic", "")) or "Not set"),
            ("Language", str(load_config().get("twitter_language", "English"))),
            ("Dialect", str(load_config().get("twitter_dialect", "") or "Default")),
            ("Cached posts", str(len(account.get("posts", [])))),
        ],
    )

    col1, col2 = st.columns(2)

    if col1.button("Generate Tweet Preview", width="stretch"):
        try:
            provider, model = ensure_llm_selected()
            twitter = Twitter(
                account["id"],
                account["nickname"],
                account["firefox_profile"],
                topic_override,
                context_override,
                open_browser=False,
            )
            st.session_state["twitter_preview_text"] = twitter.generate_post()
            st.session_state["twitter_preview_account_id"] = account["id"]
            st.success(f"Preview generated with {provider}:{model}")
        except Exception as exc:
            st.error(str(exc))
            st.code(traceback.format_exc())

    if col2.button("Post Preview to X", width="stretch"):
        try:
            preview_text = st.session_state.get("twitter_preview_text", "").strip()
            if not preview_text:
                raise RuntimeError("Generate or edit a preview first.")

            twitter = Twitter(
                account["id"],
                account["nickname"],
                account["firefox_profile"],
                topic_override,
                context_override,
                open_browser=True,
            )
            twitter.post(preview_text)
            st.success("Tweet posted.")
        except Exception as exc:
            st.error(str(exc))
            st.code(traceback.format_exc())

    preview_text = st.text_area(
        "Tweet preview",
        value=st.session_state.get("twitter_preview_text", ""),
        height=160,
    )
    st.session_state["twitter_preview_text"] = preview_text

    recent_posts = account.get("posts", [])
    if recent_posts:
        with st.expander("Recent cached posts"):
            for post in reversed(recent_posts[-10:]):
                st.write(f'[{post.get("date", "")}] {post.get("content", "")}')


def render_youtube_studio() -> None:
    accounts = get_accounts("youtube")
    if not accounts:
        st.info("Create a YouTube account first in the Accounts page.")
        return

    # Compact header: account selector + key stats in one row
    sel_col, stat1, stat2, stat3 = st.columns([2.5, 0.5, 0.5, 0.5])
    with sel_col:
        account_id = st.selectbox(
            "YouTube account",
            options=[account["id"] for account in accounts],
            format_func=lambda selected_id: next(
                f'{account["nickname"]} — {account.get("niche", "")} ({account.get("language", "")})'
                for account in accounts
                if account["id"] == selected_id
            ),
        )
    account = next(account for account in accounts if account["id"] == account_id)

    drafts = get_youtube_drafts_for_account(account["id"])
    recoverable_runs = get_youtube_recoverable_runs_for_account(account["id"])
    preview = st.session_state.get("youtube_preview", {})
    generated = st.session_state.get("youtube_generated_video", {})
    if generated.get("account_id") == account["id"] and generated.get("video_path"):
        generated_path = os.path.abspath(generated.get("video_path", ""))
        if not any(os.path.abspath(draft.get("video_path", "")) == generated_path for draft in drafts):
            drafts = [
                {
                    "account_id": account["id"],
                    "video_path": generated_path,
                    "metadata": generated.get("metadata", {}),
                    "workspace_dir": generated.get("workspace_dir", ""),
                    "subject": generated.get("subject", ""),
                    "script": generated.get("script", ""),
                    "scene_units": list(generated.get("scene_units", []) or []),
                    "image_prompts": list(generated.get("image_prompts", []) or []),
                    "created_at": "Current session",
                }
            ] + drafts

    with stat1:
        st.metric("Drafts", len(drafts))
    with stat2:
        st.metric("Videos", len(account.get("videos", [])))
    with stat3:
        st.metric("Recovery", len(recoverable_runs))

    create_tab, trends_tab, preview_tab, drafts_tab, pricing_tab = st.tabs(
        ["Create", "Trends", "Preview", "Draft Library", "Pricing"]
    )

    with create_tab:
        # Overrides in a compact expander — most users use defaults
        with st.expander("Channel overrides", expanded=False):
            setup_col, context_col = st.columns([1, 1.2])
            with setup_col:
                niche_override = st.text_input("Niche", value=account.get("niche", ""), key=f"yt_niche_{account['id']}")
                language_override = st.text_input("Language", value=account.get("language", "English"), key=f"yt_lang_{account['id']}")
                dialect_override = st.text_input("Dialect", value=account.get("dialect", ""), key=f"yt_dialect_{account['id']}")
            with context_col:
                context_override = st.text_area(
                    "Character context",
                    value=account.get("character_context", ""),
                    height=140,
                    key=f"yt_ctx_{account['id']}",
                )

        script_mode = st.radio(
            "Script source",
            options=["Generate with AI", "Write manually"],
            horizontal=True,
            key=f"youtube_script_mode_{account['id']}",
        )
        manual_subject = ""
        manual_script = ""
        seed_topic = ""
        if script_mode == "Write manually":
            manual_subject = st.text_input(
                "Topic / angle (optional)",
                key=f"youtube_manual_subject_{account['id']}",
                help="If empty, derived from the script.",
            )
            manual_script = st.text_area(
                "Video script",
                key=f"youtube_manual_script_{account['id']}",
                height=200,
                help="Paste narration. The app generates metadata, images, TTS, and the final video around it.",
            )
        else:
            seed_topic = st.text_input(
                "Title / keyword (optional)",
                key=f"youtube_seed_topic_{account['id']}",
                help="Give a title or keyword and the video is generated around it. Leave empty to auto-pick a topic from the niche.",
            )

        col1, col2, col3 = st.columns([1, 1.5, 1])

        if col1.button("Preview Idea", width="stretch"):
            try:
                provider, model = ensure_llm_selected()
                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=False,
                )
                if script_mode == "Write manually":
                    if not manual_script.strip():
                        raise RuntimeError("Enter the manual script first.")
                    youtube.set_manual_script(manual_script, manual_subject)
                    subject = youtube.subject
                    script = youtube.script
                else:
                    subject = youtube.generate_topic(seed_topic=seed_topic)
                    script = youtube.generate_script()
                metadata = youtube.generate_metadata()
                image_prompts = youtube.generate_prompts()
                st.session_state["youtube_preview"] = {
                    "account_id": account["id"],
                    "subject": subject,
                    "script": script,
                    "metadata": metadata,
                    "scene_units": list(getattr(youtube, "scene_units", []) or []),
                    "image_prompts": image_prompts,
                    "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                }
                st.success(f"Preview generated with {provider}:{model}")
            except Exception as exc:
                st.error(str(exc))
                st.code(traceback.format_exc())

        if col2.button("Generate Full Video", width="stretch", type="primary"):
            youtube = None
            try:
                provider, model = ensure_llm_selected()
                progress_callback, log_capture = build_youtube_generation_monitor()
                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=False,
                )
                youtube.set_progress_callback(progress_callback)
                with st.spinner("Generating video assets..."), contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                    if script_mode == "Write manually":
                        if not manual_script.strip():
                            raise RuntimeError("Enter the manual script first.")
                        youtube.generate_video_from_existing_script(TTS(), manual_script, manual_subject)
                    else:
                        youtube.generate_video(TTS(), seed_topic=seed_topic)
                log_capture.flush()

                st.session_state["youtube_preview"] = {
                    "account_id": account["id"],
                    "subject": getattr(youtube, "subject", ""),
                    "script": getattr(youtube, "script", ""),
                    "metadata": youtube.metadata,
                    "scene_units": list(getattr(youtube, "scene_units", []) or []),
                    "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                    "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                    "video_path": youtube.video_path,
                }
                st.session_state["youtube_generated_video"] = {
                    "account_id": account["id"],
                    "video_path": youtube.video_path,
                    "metadata": youtube.metadata,
                    "subject": getattr(youtube, "subject", ""),
                    "script": getattr(youtube, "script", ""),
                    "scene_units": list(getattr(youtube, "scene_units", []) or []),
                    "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                    "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                }
                save_youtube_draft(
                    account["id"],
                    youtube.video_path,
                    youtube.metadata,
                    workspace_dir=getattr(youtube, "_workspace_dir", ""),
                    subject=getattr(youtube, "subject", ""),
                    script=getattr(youtube, "script", ""),
                    scene_units=list(getattr(youtube, "scene_units", []) or []),
                    image_prompts=list(getattr(youtube, "image_prompts", []) or []),
                )
                st.success(f"Video generated with {provider}:{model}")
            except ImageRateLimitError as exc:
                st.error(str(exc))
                st.info(
                    "The GUI stopped early because the image API is quota-limited right now. "
                    "Try again later, lower image demand, or upgrade the active image provider quota."
                )
            except Exception as exc:
                st.error(str(exc))
                if youtube is not None and getattr(youtube, "_workspace_dir", None):
                    st.info(
                        "This run's workspace was kept on disk, so you can retry the final render "
                        f'from saved assets later: {youtube._workspace_dir}'
                    )
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

        if col3.button("Upload Latest Draft", width="stretch"):
            try:
                generated_for_upload = st.session_state.get("youtube_generated_video")
                if not generated_for_upload or generated_for_upload.get("account_id") != account["id"]:
                    generated_for_upload = get_last_youtube_draft(account["id"])

                if not generated_for_upload:
                    raise RuntimeError("Generate a video for this account first.")

                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=True,
                )
                youtube.video_path = generated_for_upload["video_path"]
                youtube.metadata = generated_for_upload["metadata"]
                uploaded = youtube.upload_video()
                if uploaded:
                    st.success("Video uploaded.")
                else:
                    st.error("Upload failed.")
            except Exception as exc:
                st.error(str(exc))
                st.code(traceback.format_exc())

        if recoverable_runs:
            with st.expander(f"Recovery ({len(recoverable_runs)} unfinished run{'s' if len(recoverable_runs) != 1 else ''})"):
                selected_recovery_workspace = st.selectbox(
                    "Recoverable run",
                    options=[run["workspace_dir"] for run in recoverable_runs],
                    format_func=lambda path: format_recoverable_run_option(
                        next(run for run in recoverable_runs if run["workspace_dir"] == path)
                    ),
                    key=f"youtube_recoverable_run_{account['id']}",
                )
                selected_recovery_run = next(
                    run for run in recoverable_runs if run["workspace_dir"] == selected_recovery_workspace
                )
                st.caption(f"Subject: {selected_recovery_run.get('subject', '(untitled)')}")
                if st.button(
                    "Retry Final Render",
                    width="stretch",
                    key=f"youtube_retry_render_{account['id']}",
                ):
                    try:
                        progress_callback, log_capture = build_youtube_generation_monitor()
                        youtube = YouTube(
                            account["id"],
                            account["nickname"],
                            account["firefox_profile"],
                            niche_override,
                            language_override,
                            dialect_override,
                            context_override,
                            account.get("is_for_kids"),
                            open_browser=False,
                        )
                        youtube.set_progress_callback(progress_callback)
                        with st.spinner("Rebuilding final video from saved assets..."), contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                            youtube.load_workspace_state(selected_recovery_workspace)
                            youtube.rerender_video_from_workspace()
                        log_capture.flush()

                        st.session_state["youtube_preview"] = {
                            "account_id": account["id"],
                            "subject": getattr(youtube, "subject", ""),
                            "script": getattr(youtube, "script", ""),
                            "metadata": youtube.metadata,
                            "scene_units": list(getattr(youtube, "scene_units", []) or []),
                            "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                            "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                            "video_path": youtube.video_path,
                        }
                        st.session_state["youtube_generated_video"] = {
                            "account_id": account["id"],
                            "video_path": youtube.video_path,
                            "metadata": youtube.metadata,
                            "subject": getattr(youtube, "subject", ""),
                            "script": getattr(youtube, "script", ""),
                            "scene_units": list(getattr(youtube, "scene_units", []) or []),
                            "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                            "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                        }
                        save_youtube_draft(
                            account["id"],
                            youtube.video_path,
                            youtube.metadata,
                            workspace_dir=getattr(youtube, "_workspace_dir", ""),
                            subject=getattr(youtube, "subject", ""),
                            script=getattr(youtube, "script", ""),
                            scene_units=list(getattr(youtube, "scene_units", []) or []),
                            image_prompts=list(getattr(youtube, "image_prompts", []) or []),
                        )
                        st.success("Final video rebuilt from the saved workspace.")
                    except Exception as exc:
                        st.error(str(exc))
                        with st.expander("Technical details", expanded=False):
                            st.code(traceback.format_exc())

    with trends_tab:
        st.caption(
            "Find trending / high-engagement titles, then send one to the Create tab as the "
            "video seed. Reddit needs no key; YouTube needs `youtube_data_api_key` in config.json."
        )

        notice = st.session_state.pop("_trend_seed_notice", "")
        if notice:
            st.success(f"Applied to Create tab (title seed + Niche override): “{notice}”. Open **Create** and click **Generate Full Video**.")

        category_options = ["Custom"] + list_trend_categories()
        trend_col1, trend_col2 = st.columns([1.4, 1])
        with trend_col1:
            selected_category = st.selectbox(
                "Category",
                options=category_options,
                index=1 if len(category_options) > 1 else 0,
                key=f"yt_trend_category_{account['id']}",
                help="Pick a suggested niche, or choose Custom to enter your own subreddits/keywords.",
            )
        with trend_col2:
            trend_sources = st.multiselect(
                "Sources",
                options=["Reddit", "YouTube"],
                default=["Reddit"],
                key=f"yt_trend_sources_{account['id']}",
            )

        custom_subreddits: list[str] = []
        custom_keywords = ""
        if selected_category == "Custom":
            subs_raw = st.text_input(
                "Subreddits (comma-separated, without r/)",
                key=f"yt_trend_subs_{account['id']}",
                help="e.g. space, askscience, todayilearned",
            )
            custom_keywords = st.text_input(
                "YouTube search keywords",
                key=f"yt_trend_kw_{account['id']}",
                help="e.g. space science facts",
            )
            custom_subreddits = [part.strip() for part in subs_raw.split(",") if part.strip()]

        region_col, mode_col = st.columns([1, 1.3])
        with region_col:
            region = st.text_input(
                "YouTube region code",
                value=load_config().get("trends_region", "US"),
                key=f"yt_trend_region_{account['id']}",
                help="ISO code (US, EG, GB, ...). Used for YouTube trends only.",
            )
        with mode_col:
            youtube_mode_label = st.selectbox(
                "YouTube mode",
                options=["Niche keyword search", "Trending in region"],
                index=0,
                key=f"yt_trend_ytmode_{account['id']}",
                help="'Niche keyword search' respects the selected Category (recommended). 'Trending in region' returns what's trending in that country — it only filters by Category when YouTube has a matching one, otherwise all topics.",
            )
        youtube_mode = "region" if youtube_mode_label == "Trending in region" else "search"
        # Bias niche search toward the account's language (e.g. Arabic -> ar).
        relevance_language = language_to_code(account.get("language", ""))

        if st.button("Fetch trending ideas", key=f"yt_trend_fetch_{account['id']}", type="primary"):
            sources = tuple(source.lower() for source in trend_sources) or ("reddit",)
            with st.spinner("Fetching trending ideas..."):
                try:
                    ideas, notes = get_trending_ideas(
                        category=None if selected_category == "Custom" else selected_category,
                        custom_subreddits=custom_subreddits or None,
                        keywords=custom_keywords or None,
                        sources=sources,
                        region=region.strip() or None,
                        language=relevance_language or None,
                        youtube_mode=youtube_mode,
                    )
                    st.session_state[f"yt_trends_{account['id']}"] = {"ideas": ideas, "notes": notes}
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state[f"yt_trends_{account['id']}"] = {"ideas": [], "notes": []}

        trend_state = st.session_state.get(f"yt_trends_{account['id']}", {})
        for note in trend_state.get("notes", []):
            st.info(note)

        ideas = trend_state.get("ideas", [])
        if ideas:
            st.write(f"**{len(ideas)} ideas** — click *Use as title* to seed the Create tab:")
            for index, idea in enumerate(ideas):
                idea_col, btn_col = st.columns([5, 1])
                with idea_col:
                    st.markdown(f"**{idea['title']}**")
                    st.caption(f"{idea['source']} · {idea.get('metric_label', '')} · [open]({idea['url']})")
                with btn_col:
                    st.button(
                        "Use as title",
                        key=f"yt_trend_use_{account['id']}_{index}",
                        on_click=_send_trend_to_seed,
                        args=(account["id"], idea["title"]),
                    )
        elif trend_state:
            st.warning("No ideas found. Try another category, add the YouTube source, or use Custom subreddits.")

    with preview_tab:
        # Re-read session state here: the Create-tab handlers run earlier in this
        # same script run and may have just set the preview, so the values
        # captured at the top of the function would be stale.
        preview = st.session_state.get("youtube_preview", {})
        generated = st.session_state.get("youtube_generated_video", {})
        preview_payload = preview if preview.get("account_id") == account["id"] else {}
        generated_payload = generated if generated.get("account_id") == account["id"] else {}
        preview_source_payload = preview_payload or generated_payload

        if preview_source_payload:
            editor_payload, editor_action = render_youtube_scene_editor(
                editor_key=f"yt_preview_editor_{account['id']}",
                source_payload=preview_source_payload,
                title="Preview Studio",
                body="Adjust the transcript, refine scene prompts, and decide whether to save the edit, generate a new full video, or rebuild from the current workspace.",
                show_video=True,
                allow_generate_from_editor=True,
            )
            editor_action_type = editor_action.get("type") if isinstance(editor_action, dict) else ""

            if editor_action_type == "save":
                try:
                    if editor_payload.get("workspace_dir"):
                        youtube = YouTube(
                            account["id"],
                            account["nickname"],
                            account["firefox_profile"],
                            niche_override,
                            language_override,
                            dialect_override,
                            context_override,
                            account.get("is_for_kids"),
                            open_browser=False,
                        )
                        youtube.apply_workspace_edits(
                            editor_payload.get("workspace_dir"),
                            subject=editor_payload.get("subject", ""),
                            script=editor_payload.get("script", ""),
                            metadata=editor_payload.get("metadata", {}),
                            scene_units=editor_payload.get("scene_units", []),
                            image_prompts=editor_payload.get("image_prompts", []),
                        )

                    preview_state = {
                        "account_id": account["id"],
                        "subject": editor_payload.get("subject", ""),
                        "script": editor_payload.get("script", ""),
                        "metadata": editor_payload.get("metadata", {}),
                        "scene_units": editor_payload.get("scene_units", []),
                        "image_prompts": editor_payload.get("image_prompts", []),
                        "workspace_dir": editor_payload.get("workspace_dir", ""),
                        "video_path": editor_payload.get("video_path", ""),
                    }
                    st.session_state["youtube_preview"] = preview_state
                    if editor_payload.get("video_path"):
                        st.session_state["youtube_generated_video"] = {
                            **preview_state,
                            "video_path": editor_payload.get("video_path", ""),
                        }
                        save_youtube_draft(
                            account["id"],
                            editor_payload.get("video_path", ""),
                            editor_payload.get("metadata", {}),
                            workspace_dir=editor_payload.get("workspace_dir", ""),
                            subject=editor_payload.get("subject", ""),
                            script=editor_payload.get("script", ""),
                            scene_units=editor_payload.get("scene_units", []),
                            image_prompts=editor_payload.get("image_prompts", []),
                        )
                    st.session_state.pop(f"yt_preview_editor_{account['id']}", None)
                    st.success("Preview edits saved.")
                except Exception as exc:
                    st.error(str(exc))
                    with st.expander("Technical details", expanded=False):
                        st.code(traceback.format_exc())

            elif editor_action_type in {"generate_from_editor", "regenerate_voiceover", "regenerate_scenes"}:
                youtube = None
                try:
                    progress_callback, log_capture = build_youtube_generation_monitor()
                    youtube = YouTube(
                        account["id"],
                        account["nickname"],
                        account["firefox_profile"],
                        niche_override,
                        language_override,
                        dialect_override,
                        context_override,
                        account.get("is_for_kids"),
                        open_browser=False,
                    )
                    youtube.set_progress_callback(progress_callback)

                    with st.spinner("Applying editor changes..."), contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                        if editor_action_type == "generate_from_editor":
                            youtube.generate_video_from_editor(
                                TTS(),
                                editor_payload.get("script", ""),
                                editor_payload.get("subject", ""),
                                metadata=editor_payload.get("metadata", {}),
                                scene_units=editor_payload.get("scene_units", []),
                                image_prompts=editor_payload.get("image_prompts", []),
                            )
                        elif editor_action_type in {"regenerate_voiceover", "regenerate_scenes"}:
                            youtube.apply_workspace_edits(
                                editor_payload.get("workspace_dir", ""),
                                subject=editor_payload.get("subject", ""),
                                script=editor_payload.get("script", ""),
                                metadata=editor_payload.get("metadata", {}),
                                scene_units=editor_payload.get("scene_units", []),
                                image_prompts=editor_payload.get("image_prompts", []),
                            )
                            youtube.rebuild_video_from_workspace(
                                TTS(),
                                editor_payload.get("workspace_dir", ""),
                                regenerate_voiceover=True,
                                regenerate_assets=editor_action_type == "regenerate_scenes",
                            )
                    log_capture.flush()

                    st.session_state["youtube_preview"] = {
                        "account_id": account["id"],
                        "subject": getattr(youtube, "subject", ""),
                        "script": getattr(youtube, "script", ""),
                        "metadata": youtube.metadata,
                        "scene_units": list(getattr(youtube, "scene_units", []) or []),
                        "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                        "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                        "video_path": youtube.video_path,
                    }
                    st.session_state["youtube_generated_video"] = {
                        "account_id": account["id"],
                        "video_path": youtube.video_path,
                        "metadata": youtube.metadata,
                        "subject": getattr(youtube, "subject", ""),
                        "script": getattr(youtube, "script", ""),
                        "scene_units": list(getattr(youtube, "scene_units", []) or []),
                        "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                        "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                    }
                    save_youtube_draft(
                        account["id"],
                        youtube.video_path,
                        youtube.metadata,
                        workspace_dir=getattr(youtube, "_workspace_dir", ""),
                        subject=getattr(youtube, "subject", ""),
                        script=getattr(youtube, "script", ""),
                        scene_units=list(getattr(youtube, "scene_units", []) or []),
                        image_prompts=list(getattr(youtube, "image_prompts", []) or []),
                    )
                    st.session_state.pop(f"yt_preview_editor_{account['id']}", None)
                    if editor_action_type == "generate_from_editor":
                        st.success("Generated a new full video from the edited preview.")
                    elif editor_action_type == "regenerate_voiceover":
                        st.success("Voiceover and final video were rebuilt from the edited draft.")
                    else:
                        st.success("Scenes, voiceover, and final video were rebuilt from the edited draft.")
                except Exception as exc:
                    st.error(str(exc))
                    if youtube is not None and getattr(youtube, "_workspace_dir", None):
                        st.info(f'Workspace kept at: {youtube._workspace_dir}')
                    with st.expander("Technical details", expanded=False):
                        st.code(traceback.format_exc())
            elif editor_action_type == "regenerate_scene_asset":
                try:
                    youtube = YouTube(
                        account["id"],
                        account["nickname"],
                        account["firefox_profile"],
                        niche_override,
                        language_override,
                        dialect_override,
                        context_override,
                        account.get("is_for_kids"),
                        open_browser=False,
                    )
                    youtube.apply_workspace_edits(
                        editor_payload.get("workspace_dir", ""),
                        subject=editor_payload.get("subject", ""),
                        script=editor_payload.get("script", ""),
                        metadata=editor_payload.get("metadata", {}),
                        scene_units=editor_payload.get("scene_units", []),
                        image_prompts=editor_payload.get("image_prompts", []),
                    )
                    if str(editor_action.get("provider", "")).lower() == "pixabay":
                        picker_payload = youtube.list_pixabay_scene_candidates(
                            int(editor_action.get("scene_index", 0) or 0),
                            editor_payload.get("workspace_dir", ""),
                        )
                        st.session_state[YOUTUBE_PIXABAY_PICKER_STATE_KEY] = {
                            **picker_payload,
                            "account_id": account["id"],
                            "editor_key": f"yt_preview_editor_{account['id']}",
                            "editor_payload": editor_payload,
                        }
                    else:
                        youtube.regenerate_scene_asset(
                            int(editor_action.get("scene_index", 0) or 0),
                            str(editor_action.get("provider", "nanobanana2") or "nanobanana2"),
                            editor_payload.get("workspace_dir", ""),
                        )
                        persist_youtube_editor_result(account["id"], youtube)
                        st.session_state.pop(f"yt_preview_editor_{account['id']}", None)
                    if str(editor_action.get("provider", "")).lower() == "pixabay":
                        st.info("Choose the Pixabay asset you want from the popup.")
                    else:
                        st.success(
                            f"Scene {int(editor_action.get('scene_index', 0) or 0) + 1} asset regenerated via Nano Banana."
                        )
                except Exception as exc:
                    st.error(str(exc))
                    with st.expander("Technical details", expanded=False):
                        st.code(traceback.format_exc())
        else:
            st.info("No preview yet. Generate an idea/script preview from the Create tab.")

    with drafts_tab:
        if drafts:
            # Draft selector + upload + cost in one compact row
            draft_sel_col, draft_upload_col, draft_cost_col = st.columns([2, 0.8, 0.6])
            with draft_sel_col:
                selected_draft_path = st.selectbox(
                    "Draft",
                    options=[draft["video_path"] for draft in drafts],
                    format_func=lambda path: format_draft_option(
                        next(draft for draft in drafts if draft["video_path"] == path)
                    ),
                    key=f"youtube_draft_select_{account['id']}",
                    label_visibility="collapsed",
                )
            selected_draft = next(draft for draft in drafts if draft["video_path"] == selected_draft_path)
            selected_metadata = selected_draft.get("metadata", {})
            selected_pricing = load_pricing_for_video(selected_draft_path, selected_metadata)
            with draft_cost_col:
                if selected_pricing:
                    cost_val = float(selected_pricing.get('total_estimated_cost', 0.0) or 0.0)
                    st.metric("Cost", f"${cost_val:.3f}")
                else:
                    st.metric("Cost", "N/A")

            editor_payload, editor_action = render_youtube_scene_editor(
                editor_key=f"yt_draft_editor_{account['id']}",
                source_payload=selected_draft,
                title="Draft Editor",
                body="",
                show_video=True,
                allow_generate_from_editor=False,
            )
            editor_action_type = editor_action.get("type") if isinstance(editor_action, dict) else ""

            if draft_upload_col.button("Upload", width="stretch", key=f"upload_selected_draft_{account['id']}", type="primary"):
                try:
                    youtube = YouTube(
                        account["id"],
                        account["nickname"],
                        account["firefox_profile"],
                        account.get("niche", ""),
                        account.get("language", "English"),
                        account.get("dialect", ""),
                        account.get("character_context", ""),
                        account.get("is_for_kids"),
                        open_browser=True,
                    )
                    youtube.video_path = selected_draft_path
                    youtube.metadata = editor_payload.get("metadata", {})
                    if youtube.upload_video():
                        st.success("Video uploaded.")
                    else:
                        st.error("Upload failed.")
                except Exception as exc:
                    st.error(str(exc))
                    with st.expander("Technical details", expanded=False):
                        st.code(traceback.format_exc())

            if editor_action:
                try:
                    if editor_action_type == "save":
                        if editor_payload.get("workspace_dir"):
                            youtube = YouTube(
                                account["id"],
                                account["nickname"],
                                account["firefox_profile"],
                                niche_override,
                                language_override,
                                dialect_override,
                                context_override,
                                account.get("is_for_kids"),
                                open_browser=False,
                            )
                            youtube.apply_workspace_edits(
                                editor_payload.get("workspace_dir", ""),
                                subject=editor_payload.get("subject", ""),
                                script=editor_payload.get("script", ""),
                                metadata=editor_payload.get("metadata", {}),
                                scene_units=editor_payload.get("scene_units", []),
                                image_prompts=editor_payload.get("image_prompts", []),
                            )

                        save_youtube_draft(
                            account["id"],
                            editor_payload.get("video_path", selected_draft_path),
                            editor_payload.get("metadata", {}),
                            workspace_dir=editor_payload.get("workspace_dir", ""),
                            subject=editor_payload.get("subject", ""),
                            script=editor_payload.get("script", ""),
                            scene_units=editor_payload.get("scene_units", []),
                            image_prompts=editor_payload.get("image_prompts", []),
                        )
                        st.session_state.pop(f"yt_draft_editor_{account['id']}", None)
                        st.success("Draft edits saved.")
                    elif editor_action_type in {"regenerate_voiceover", "regenerate_scenes"}:
                        progress_callback, log_capture = build_youtube_generation_monitor()
                        youtube = YouTube(
                            account["id"],
                            account["nickname"],
                            account["firefox_profile"],
                            niche_override,
                            language_override,
                            dialect_override,
                            context_override,
                            account.get("is_for_kids"),
                            open_browser=False,
                        )
                        youtube.set_progress_callback(progress_callback)
                        with st.spinner("Rebuilding edited draft..."), contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                            youtube.apply_workspace_edits(
                                editor_payload.get("workspace_dir", ""),
                                subject=editor_payload.get("subject", ""),
                                script=editor_payload.get("script", ""),
                                metadata=editor_payload.get("metadata", {}),
                                scene_units=editor_payload.get("scene_units", []),
                                image_prompts=editor_payload.get("image_prompts", []),
                            )
                            if editor_action_type == "regenerate_scene_asset":
                                youtube.regenerate_scene_asset(
                                    int(editor_action.get("scene_index", 0) or 0),
                                    str(editor_action.get("provider", "pixabay") or "pixabay"),
                                    editor_payload.get("workspace_dir", ""),
                                )
                            else:
                                youtube.rebuild_video_from_workspace(
                                    TTS(),
                                    editor_payload.get("workspace_dir", ""),
                                    regenerate_voiceover=True,
                                    regenerate_assets=editor_action_type == "regenerate_scenes",
                                )
                        log_capture.flush()

                        st.session_state["youtube_preview"] = {
                            "account_id": account["id"],
                            "subject": getattr(youtube, "subject", ""),
                            "script": getattr(youtube, "script", ""),
                            "metadata": youtube.metadata,
                            "scene_units": list(getattr(youtube, "scene_units", []) or []),
                            "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                            "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                            "video_path": youtube.video_path,
                        }
                        st.session_state["youtube_generated_video"] = {
                            "account_id": account["id"],
                            "video_path": youtube.video_path,
                            "metadata": youtube.metadata,
                            "subject": getattr(youtube, "subject", ""),
                            "script": getattr(youtube, "script", ""),
                            "scene_units": list(getattr(youtube, "scene_units", []) or []),
                            "image_prompts": list(getattr(youtube, "image_prompts", []) or []),
                            "workspace_dir": getattr(youtube, "_workspace_dir", ""),
                        }
                        save_youtube_draft(
                            account["id"],
                            youtube.video_path,
                            youtube.metadata,
                            workspace_dir=getattr(youtube, "_workspace_dir", ""),
                            subject=getattr(youtube, "subject", ""),
                            script=getattr(youtube, "script", ""),
                            scene_units=list(getattr(youtube, "scene_units", []) or []),
                            image_prompts=list(getattr(youtube, "image_prompts", []) or []),
                        )
                        st.session_state.pop(f"yt_draft_editor_{account['id']}", None)
                        if editor_action_type == "regenerate_voiceover":
                            st.success("Draft voiceover and video rebuilt.")
                        else:
                            st.success("Draft scenes, voiceover, and video rebuilt.")
                    elif editor_action_type == "regenerate_scene_asset":
                        youtube = YouTube(
                            account["id"],
                            account["nickname"],
                            account["firefox_profile"],
                            niche_override,
                            language_override,
                            dialect_override,
                            context_override,
                            account.get("is_for_kids"),
                            open_browser=False,
                        )
                        youtube.apply_workspace_edits(
                            editor_payload.get("workspace_dir", ""),
                            subject=editor_payload.get("subject", ""),
                            script=editor_payload.get("script", ""),
                            metadata=editor_payload.get("metadata", {}),
                            scene_units=editor_payload.get("scene_units", []),
                            image_prompts=editor_payload.get("image_prompts", []),
                        )
                        if str(editor_action.get("provider", "")).lower() == "pixabay":
                            picker_payload = youtube.list_pixabay_scene_candidates(
                                int(editor_action.get("scene_index", 0) or 0),
                                editor_payload.get("workspace_dir", ""),
                            )
                            st.session_state[YOUTUBE_PIXABAY_PICKER_STATE_KEY] = {
                                **picker_payload,
                                "account_id": account["id"],
                                "editor_key": f"yt_draft_editor_{account['id']}",
                                "editor_payload": editor_payload,
                            }
                        else:
                            youtube.regenerate_scene_asset(
                                int(editor_action.get("scene_index", 0) or 0),
                                str(editor_action.get("provider", "nanobanana2") or "nanobanana2"),
                                editor_payload.get("workspace_dir", ""),
                            )
                            persist_youtube_editor_result(account["id"], youtube)
                            st.session_state.pop(f"yt_draft_editor_{account['id']}", None)
                        if str(editor_action.get("provider", "")).lower() == "pixabay":
                            st.info("Choose the Pixabay asset you want from the popup.")
                        else:
                            st.success(
                                f"Scene {int(editor_action.get('scene_index', 0) or 0) + 1} asset regenerated via Nano Banana."
                            )
                except Exception as exc:
                    st.error(str(exc))
                    with st.expander("Technical details", expanded=False):
                        st.code(traceback.format_exc())
        else:
            st.info("No saved drafts yet. Generate a video or import an existing MP4 to start building a draft library.")

        with st.expander("Import Existing Video File"):
            with st.form(f"import_video_{account['id']}"):
                import_video_path = st.text_input(
                    "MP4 path",
                    value="",
                    help="Use this to recover a previously generated MP4 and make it uploadable from the GUI.",
                )
                import_title = st.text_input("Title", value=preview.get("metadata", {}).get("title", ""))
                import_description = st.text_area(
                    "Description",
                    value=preview.get("metadata", {}).get("description", ""),
                    height=120,
                )
                import_button = st.form_submit_button("Save As Draft", width="stretch")

            if import_button:
                normalized_import_path = os.path.abspath(import_video_path.strip())
                if not import_video_path.strip():
                    st.error("Enter the MP4 path first.")
                elif not os.path.exists(normalized_import_path):
                    st.error("That MP4 path does not exist.")
                else:
                    save_youtube_draft(
                        account["id"],
                        normalized_import_path,
                        {
                            "title": import_title.strip(),
                            "description": import_description.strip(),
                        },
                        subject=preview.get("subject", ""),
                        script=preview.get("script", ""),
                        scene_units=list(preview.get("scene_units", []) or []),
                        image_prompts=list(preview.get("image_prompts", []) or []),
                    )
                    st.success("Video draft saved. You can now use Upload Latest Draft.")
                    st.rerun()

        recent_videos = account.get("videos", [])
        if recent_videos:
            with st.expander("Recent cached uploads"):
                for video in reversed(recent_videos[-10:]):
                    st.write(f'[{video.get("date", "")}] {video.get("title", "")}')
                    st.write(video.get("url", ""))

    with pricing_tab:
        st.markdown("#### Pricing Ledger")
        current_generated = st.session_state.get("youtube_generated_video", {})
        generated_path = ""
        generated_metadata = {}
        if current_generated.get("account_id") == account["id"]:
            generated_path = current_generated.get("video_path", "")
            generated_metadata = current_generated.get("metadata", {})

        source_options: list[tuple[str, str, dict]] = []
        if generated_path:
            source_options.append(("Current session video", generated_path, generated_metadata))
        for draft in drafts:
            source_options.append((format_draft_option(draft), draft.get("video_path", ""), draft.get("metadata", {})))

        deduped_options = []
        seen_paths = set()
        for label, path, metadata in source_options:
            normalized = os.path.abspath(str(path or ""))
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            deduped_options.append((label, normalized, metadata))

        if deduped_options:
            selected_cost_path = st.selectbox(
                "Choose a generated video",
                options=[path for _, path, _ in deduped_options],
                format_func=lambda selected_path: next(
                    label for label, path, _ in deduped_options if path == selected_path
                ),
                key=f"youtube_pricing_select_{account['id']}",
            )
            selected_cost_metadata = next(
                metadata for _, path, metadata in deduped_options if path == selected_cost_path
            )
            render_panel(
                "Estimated Spend",
                "This report sums text generation, image generation, TTS, and subtitle processing for the selected video run. Free assets still appear so you can see what cost zero.",
                soft=True,
            )
            render_pricing_report(
                load_pricing_for_video(selected_cost_path, selected_cost_metadata),
                key_prefix=f"youtube_pricing_{account['id']}",
            )
        else:
            st.info("Generate or save a draft first to see its pricing breakdown.")

    picker_state = st.session_state.get(YOUTUBE_PIXABAY_PICKER_STATE_KEY, {})
    if isinstance(picker_state, dict) and picker_state.get("account_id") == account["id"]:
        render_youtube_pixabay_picker_dialog(
            account,
            niche_override,
            language_override,
            dialect_override,
            context_override,
        )


def render_tiktok_studio() -> None:
    accounts = get_accounts("tiktok")
    if not accounts:
        st.info("Create a TikTok account first in the Accounts page.")
        return

    sel_col, stat_col = st.columns([3, 0.5])
    with sel_col:
        account_id = st.selectbox(
            "TikTok account",
            options=[account["id"] for account in accounts],
            format_func=lambda selected_id: next(
                f'{account["nickname"]} — {account.get("niche", "")} ({account.get("language", "")})'
                for account in accounts
                if account["id"] == selected_id
            ),
            key="tiktok_studio_account_select",
        )
    account = next(account for account in accounts if account["id"] == account_id)
    with stat_col:
        st.metric("Uploads", len(account.get("videos", [])))

    with st.expander("Channel overrides", expanded=False):
        ov_left, ov_right = st.columns([1, 1.2])
        with ov_left:
            niche_override = st.text_input("Niche", value=account.get("niche", ""), key=f"tiktok_niche_override_{account['id']}")
            language_override = st.text_input("Language", value=account.get("language", "English"), key=f"tiktok_language_override_{account['id']}")
            dialect_override = st.text_input("Dialect", value=account.get("dialect", ""), key=f"tiktok_dialect_override_{account['id']}")
        with ov_right:
            context_override = st.text_area(
                "Character context",
                value=account.get("character_context", ""),
                key=f"tiktok_context_override_{account['id']}",
                height=140,
            )

    script_mode = st.radio(
        "Script source",
        options=["Generate with AI", "Write manually"],
        horizontal=True,
        key=f"tiktok_script_mode_{account['id']}",
    )
    manual_subject = ""
    manual_script = ""
    seed_topic = ""
    if script_mode == "Write manually":
        manual_subject = st.text_input(
            "Topic / angle (optional)",
            key=f"tiktok_manual_subject_{account['id']}",
            help="If left empty, derived from the script.",
        )
        manual_script = st.text_area(
            "Video script",
            key=f"tiktok_manual_script_{account['id']}",
            height=200,
        )
    else:
        seed_topic = st.text_input(
            "Title / keyword (optional)",
            key=f"tiktok_seed_topic_{account['id']}",
            help="Give a title or keyword and the video is generated around it. Leave empty to auto-pick a topic from the niche.",
        )

    render_kv_card(
        "TikTok Workspace",
        account.get("nickname", "TikTok account"),
        [
            ("Niche", niche_override or "Not set"),
            ("Language", language_override or "Not set"),
            ("Dialect", dialect_override or "Default"),
            ("Cached uploads", str(len(account.get("videos", [])))),
        ],
    )

    preview_state_key = f"tiktok_video_preview_{account['id']}"
    generated_state_key = f"tiktok_generated_video_{account['id']}"
    source_mode = st.radio(
        "Video source",
        options=["Use YouTube draft", "Generate new video here"],
        horizontal=True,
        key=f"tiktok_source_mode_{account['id']}",
    )

    youtube_drafts = get_all_youtube_drafts()
    source_path = ""
    source_metadata = {}

    if source_mode == "Generate new video here":
        action_col1, action_col2 = st.columns(2)

        if action_col1.button(
            "Generate Idea + Script Preview",
            width="stretch",
            key=f"tiktok_generate_preview_{account['id']}",
        ):
            try:
                provider, model = ensure_llm_selected()
                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=False,
                )
                if script_mode == "Write manually":
                    if not manual_script.strip():
                        raise RuntimeError("Enter the manual script first.")
                    youtube.set_manual_script(manual_script, manual_subject)
                    subject = youtube.subject
                    script = youtube.script
                else:
                    subject = youtube.generate_topic(seed_topic=seed_topic)
                    script = youtube.generate_script()
                metadata = youtube.generate_metadata()
                st.session_state[preview_state_key] = {
                    "account_id": account["id"],
                    "subject": subject,
                    "script": script,
                    "metadata": metadata,
                }
                st.success(f"Preview generated with {provider}:{model}")
            except Exception as exc:
                st.error(str(exc))
                with st.expander("Technical details", expanded=False):
                    st.code(traceback.format_exc())

        if action_col2.button(
            "Generate Full Video For TikTok",
            width="stretch",
            key=f"tiktok_generate_video_{account['id']}",
        ):
            try:
                provider, model = ensure_llm_selected()
                youtube = YouTube(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    account.get("is_for_kids"),
                    open_browser=False,
                )
                with st.spinner("Generating TikTok video assets..."):
                    if script_mode == "Write manually":
                        if not manual_script.strip():
                            raise RuntimeError("Enter the manual script first.")
                        youtube.generate_video_from_existing_script(TTS(), manual_script, manual_subject)
                    else:
                        youtube.generate_video(TTS(), seed_topic=seed_topic)

                st.session_state[generated_state_key] = {
                    "account_id": account["id"],
                    "video_path": youtube.video_path,
                    "metadata": youtube.metadata,
                }
                save_youtube_draft(account["id"], youtube.video_path, youtube.metadata)
                st.success(f"Video generated with {provider}:{model}")
            except ImageRateLimitError as exc:
                st.error(str(exc))
                st.info(
                    "The image provider is quota-limited right now. Try again later, lower image demand, or switch providers."
                )
            except Exception as exc:
                st.error(str(exc))
                with st.expander("Technical details", expanded=False):
                    st.code(traceback.format_exc())

        preview = st.session_state.get(preview_state_key, {})
        if preview.get("account_id") == account["id"]:
            with st.expander("Generated preview", expanded=False):
                st.text_input(
                    "Topic",
                    value=preview.get("subject", ""),
                    disabled=True,
                    key=f"tiktok_preview_subject_{account['id']}",
                )
                st.text_area(
                    "Script",
                    value=preview.get("script", ""),
                    height=180,
                    disabled=True,
                    key=f"tiktok_preview_script_{account['id']}",
                )
                preview_metadata = preview.get("metadata", {})
                st.text_input(
                    "Title",
                    value=preview_metadata.get("title", ""),
                    disabled=True,
                    key=f"tiktok_preview_title_{account['id']}",
                )
                st.text_area(
                    "Description",
                    value=preview_metadata.get("description", ""),
                    height=120,
                    disabled=True,
                    key=f"tiktok_preview_description_{account['id']}",
                )

        generated = st.session_state.get(generated_state_key, {})
        if generated.get("account_id") == account["id"] and generated.get("video_path"):
            generated_path = os.path.abspath(generated.get("video_path", ""))
            if os.path.exists(generated_path):
                source_path = generated_path
                source_metadata = generated.get("metadata", {})
                st.success("Using the video generated in TikTok Studio for this upload.")
            else:
                st.warning("The last generated TikTok Studio video is no longer available on disk.")
        else:
            st.info("Generate a full video here to use it as the TikTok source, or switch to 'Use YouTube draft'.")
    else:
        if youtube_drafts:
            selected_source = st.selectbox(
                "Source video",
                options=[draft["video_path"] for draft in youtube_drafts],
                format_func=lambda path: format_draft_option(
                    next(draft for draft in youtube_drafts if draft["video_path"] == path)
                ),
                key=f"tiktok_source_draft_select_{account['id']}",
            )
            selected_draft = next(draft for draft in youtube_drafts if draft["video_path"] == selected_source)
            source_path = selected_draft["video_path"]
            source_metadata = selected_draft.get("metadata", {})
        else:
            st.info("No YouTube drafts found yet. You can still paste a video path manually below.")

    manual_source_path = st.text_input(
        "Or paste a video path manually",
        value="",
        help="Optional override if you want to upload a video that is not in the YouTube draft library.",
        key=f"tiktok_manual_source_path_{account['id']}",
    ).strip()
    if manual_source_path:
        source_path = os.path.abspath(manual_source_path)
        source_metadata = {}

    tiktok_helper = TikTok(
        account["id"],
        account["nickname"],
        account["firefox_profile"],
        niche_override,
        language_override,
        dialect_override,
        context_override,
        open_browser=False,
    )

    caption_state_key = f"tiktok_caption_preview_{account['id']}"
    caption_editor_key = f"tiktok_caption_editor_{account['id']}"
    source_signature_key = f"tiktok_source_signature_{account['id']}"
    current_source_signature = json.dumps(
        {
            "mode": source_mode,
            "path": os.path.abspath(source_path) if source_path else "",
            "title": source_metadata.get("title", ""),
            "description": source_metadata.get("description", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    default_caption = tiktok_helper.build_basic_caption(source_metadata) if source_metadata else ""

    if st.session_state.get(source_signature_key) != current_source_signature:
        st.session_state[source_signature_key] = current_source_signature
        st.session_state[caption_state_key] = default_caption
        st.session_state[caption_editor_key] = default_caption
    else:
        default_caption = st.session_state.get(caption_state_key, default_caption)

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Build Caption From Metadata", width="stretch"):
        rebuilt_caption = tiktok_helper.build_basic_caption(source_metadata)
        st.session_state[caption_state_key] = rebuilt_caption
        st.session_state[caption_editor_key] = rebuilt_caption

    if action_col2.button("Refine Caption With AI", width="stretch"):
        try:
            provider, model = ensure_llm_selected()
            refined = tiktok_helper.generate_caption(source_metadata)
            st.session_state[caption_state_key] = refined
            st.session_state[caption_editor_key] = refined
            st.success(f"TikTok caption refined with {provider}:{model}")
        except Exception as exc:
            st.error(str(exc))
            with st.expander("Technical details", expanded=False):
                st.code(traceback.format_exc())

    caption_preview = st.text_area(
        "TikTok caption",
        value=st.session_state.get(caption_state_key, default_caption),
        height=150,
        key=caption_editor_key,
    )
    st.session_state[caption_state_key] = caption_preview

    preview_col, detail_col = st.columns([1.35, 1])
    with preview_col:
        st.markdown(
            f"""
            <div class="studio-card">
                <div class="studio-eyebrow">Upload Target</div>
                <div class="studio-title" dir="auto">{escape(account["nickname"])}</div>
                <div class="studio-subtitle"><strong>Language:</strong> {escape(language_override or account.get("language", "English"))}</div>
                <div class="studio-subtitle"><strong>Dialect:</strong> {escape(dialect_override or account.get("dialect", "") or "Default")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if source_path and os.path.exists(source_path):
            st.video(source_path)
            st.markdown(f'<div class="studio-path">{escape(source_path)}</div>', unsafe_allow_html=True)
        else:
            st.info("Choose or paste a valid source video to preview it here.")

    with detail_col:
        st.markdown(
            f"""
            <div class="studio-card">
                <div class="studio-eyebrow">Source Metadata</div>
                <div class="studio-title" dir="auto">{escape(str(source_metadata.get("title", "(untitled)")))}</div>
                <div class="studio-subtitle">{escape(str(source_metadata.get("description", "")))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_chip_row("Hashtags", source_metadata.get("hashtags", []))
        render_chip_row("Tags", source_metadata.get("tags", []), muted=True)
        source_pricing = load_pricing_for_video(source_path, source_metadata) if source_path else {}
        if source_pricing:
            with st.expander("Source Video Pricing", expanded=False):
                render_pricing_report(
                    source_pricing,
                    key_prefix=f"tiktok_source_pricing_{account['id']}",
                )

        if st.button("Upload To TikTok", width="stretch", key="tiktok_upload_button"):
            try:
                if not source_path or not os.path.exists(source_path):
                    raise RuntimeError("Choose a valid source video before uploading to TikTok.")

                uploader = TikTok(
                    account["id"],
                    account["nickname"],
                    account["firefox_profile"],
                    niche_override,
                    language_override,
                    dialect_override,
                    context_override,
                    open_browser=True,
                )
                uploader.upload_video(source_path, caption_preview.strip())
                st.success("Video uploaded to TikTok.")
            except Exception as exc:
                st.error(str(exc))
                with st.expander("Technical details", expanded=False):
                    st.code(traceback.format_exc())

    recent_videos = account.get("videos", [])
    if recent_videos:
        with st.expander("Recent cached TikTok uploads"):
            for video in reversed(recent_videos[-10:]):
                st.write(f'[{video.get("date", "")}] {video.get("caption", "")[:100]}')
                if video.get("url"):
                    st.write(video.get("url", ""))


PAGES = {
    "Overview": "overview",
    "Configuration": "config",
    "Accounts": "accounts",
    "YouTube Studio": "youtube_studio",
    "Twitter Studio": "twitter_studio",
    "TikTok Studio": "tiktok_studio",
}

SIDEBAR_SECTIONS = {
    "Home": ["Overview"],
    "Create": ["YouTube Studio", "Twitter Studio", "TikTok Studio"],
    "Manage": ["Accounts", "Configuration"],
}


def render_sidebar_status() -> None:
    config = load_config()
    provider = config.get("llm_provider", "ollama")
    model = config.get("openai_model") if provider == "openai" else config.get("ollama_model")
    st.sidebar.markdown(
        f"""
        <div style="
            padding: 0.7rem 0.8rem;
            border-radius: 16px;
            background: rgba(31, 122, 90, 0.08);
            border: 1px solid rgba(31, 122, 90, 0.12);
            margin-bottom: 0.6rem;
            font-size: 0.82rem;
            line-height: 1.6;
            color: var(--studio-ink-soft);
        ">
            <strong style="color: var(--studio-ink);">Active Profile</strong><br>
            LLM: <strong>{escape(provider)}</strong> / {escape(model or "Not set")}<br>
            Images: <strong>{escape(config.get("image_provider", "nanobanana2"))}</strong> ({escape(config.get("asset_strategy", "mixed"))})<br>
            TTS: <strong>{escape(config.get("tts_provider", "auto"))}</strong><br>
            FPS: <strong>{config.get("video_fps", 30)}</strong> &middot;
            SFX: <strong>{"On" if config.get("sound_effects_enabled") else "Off"}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_accounts_page() -> None:
    render_section_intro(
        "Accounts",
        "Manage all your platform accounts in one place. Select a platform to view, edit, or create accounts.",
        eyebrow="Account Management",
    )
    platform = st.selectbox(
        "Platform",
        options=["youtube", "twitter", "tiktok"],
        format_func=lambda p: p.title(),
        key="accounts_platform_select",
    )
    render_account_editor(platform)


def main() -> None:
    st.set_page_config(
        page_title="MoneyPrinterPro Studio",
        page_icon="https://img.icons8.com/fluency/48/movie-projector.png",
        layout="wide",
    )

    inject_app_styles()

    # Sidebar navigation
    st.sidebar.markdown(
        """
        <div style="
            padding: 0.5rem 0 0.8rem;
            text-align: center;
        ">
            <div style="font-size: 1.35rem; font-weight: 800; color: var(--studio-ink); line-height: 1.2;">
                MoneyPrinterPro
            </div>
            <div style="font-size: 0.75rem; color: var(--studio-ink-soft); letter-spacing: 0.08em; text-transform: uppercase;">
                Studio
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar_status()

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "overview"

    for section_label, page_names in SIDEBAR_SECTIONS.items():
        st.sidebar.markdown(
            f'<div style="font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; '
            f'color: #668072; font-weight: 700; padding: 0.6rem 0 0.25rem 0.2rem;">'
            f'{escape(section_label)}</div>',
            unsafe_allow_html=True,
        )
        for page_name in page_names:
            page_key = PAGES[page_name]
            is_active = st.session_state["current_page"] == page_key
            if st.sidebar.button(
                page_name,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["current_page"] = page_key
                st.rerun()

    # Render selected page
    page = st.session_state.get("current_page", "overview")

    if page == "overview":
        render_overview()
    elif page == "config":
        render_config_tab()
    elif page == "accounts":
        render_accounts_page()
    elif page == "youtube_studio":
        render_youtube_studio()
    elif page == "twitter_studio":
        render_twitter_studio()
    elif page == "tiktok_studio":
        render_tiktok_studio()
    else:
        render_overview()


if __name__ == "__main__":
    main()
