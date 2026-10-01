import re
import base64
import json
import time
import os
import io
import hashlib
import tempfile
import requests
import assemblyai as aai
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import features as pil_features

try:
    import arabic_reshaper
except Exception:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display as bidi_get_display
except Exception:
    bidi_get_display = None

from utils import *
from cache import *
from .Tts import TTS, word_timings_path
from llm_provider import generate_text, generate_text_result
from config import *
from status import *
from uuid import uuid4
from constants import *
from typing import List
from moviepy.editor import *
from termcolor import colored
from proglog import ProgressBarLogger
from selenium_firefox import *
from selenium import webdriver
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from webdriver_manager.firefox import GeckoDriverManager
from datetime import datetime

if not hasattr(Image, "ANTIALIAS") and hasattr(Image, "Resampling"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})

DEFAULT_MIN_IMAGE_PROMPTS = 10
DEFAULT_MAX_IMAGE_PROMPTS = 12
IMAGE_RATE_LIMIT_RETRIES = 3
IMAGE_RATE_LIMIT_BACKOFF_SECONDS = 5
PIXBAY_MIN_SELECTION_SCORE = 65.0
STOCK_CACHE_TTL_SECONDS = 86400
HTTP_SESSION = requests.Session()
HTTP_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
# Script lines that describe visuals ("صورة المشهد: ...", "Scene: ...") and
# short "Label:" prefixes ("مفاجأة ذكية: ..."), both of which TTS would read aloud.
SCRIPT_DIRECTION_LINE = re.compile(
    r"^\s*(?:صورة المشهد|في المشهد|المشهد|مشهد|وصف المشهد|scene|visual|shot|b-roll)\s*[:：]",
    re.IGNORECASE,
)
SCRIPT_LABEL_PREFIX = re.compile(r"^\s*(?:[^\s:：.!?؟،,]+\s+){0,2}[^\s:：.!?؟،,]+\s*[:：]\s+")
PORTRAIT_VIDEO_BONUS = 8.0  # 9:16 clips avoid the heavy crop landscape footage needs
# CLIP rerank calibration: bonus = (cosine - BASELINE) * WEIGHT. ViT-B-32 puts a
# clear match near 0.30 and an unrelated image near 0.15. Tune these two
# knobs against pixabay_selection_debug.json ("clip_similarity") if picks drift.
CLIP_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")
CLIP_RERANK_TOP_N = 40  # scenes collect 12-32 candidates, so this checks them all
CLIP_SIMILARITY_BASELINE = 0.20
CLIP_SIMILARITY_WEIGHT = 150.0
CLIP_TOPIC_HANDICAP = 0.02  # prefer a scene match over an equally strong topic-only match
# Thumbnails darker than this mean luma (0-255) are mostly-black clips.
DARK_THUMBNAIL_LUMA = 45.0
DARK_THUMBNAIL_PENALTY = 25.0
VISUAL_ASSET_RETRY_ROUNDS = 2


class ImageRateLimitError(RuntimeError):
    """
    Raised when the configured image provider is rate-limited and the request
    should stop instead of continuing through the remaining prompts.
    """


def _is_retryable_rate_limit(response: requests.Response) -> bool:
    """
    Returns whether the HTTP response indicates a retryable rate-limit case.

    Args:
        response (requests.Response): provider response

    Returns:
        is_retryable (bool): True if we should stop and report rate limiting
    """
    return response.status_code == 429


def _is_retryable_server_error(response: requests.Response) -> bool:
    """
    Returns whether the response is a transient server error worth a quick
    retry (e.g. Gemini 503 "overloaded").

    Args:
        response (requests.Response): provider response

    Returns:
        is_retryable (bool): True for 500/502/503/504
    """
    return response.status_code in (500, 502, 503, 504)


def _nanobanana_response_has_no_image(body: dict) -> bool:
    """
    Returns whether Gemini/Nano Banana responded successfully but did not
    include an image payload.

    Args:
        body (dict): provider response JSON

    Returns:
        no_image (bool): True when a candidate explicitly finished with NO_IMAGE
    """
    for candidate in body.get("candidates", []):
        if str(candidate.get("finishReason", "")).upper() == "NO_IMAGE":
            return True
    return False


class YouTube:
    """
    Class for YouTube Automation.

    Steps to create a YouTube Short:
    1. Generate a topic [DONE]
    2. Generate a script [DONE]
    3. Generate metadata (Title, Description, Tags) [DONE]
    4. Generate AI Image Prompts [DONE]
    4. Generate Images based on generated Prompts [DONE]
    5. Convert Text-to-Speech [DONE]
    6. Show images each for n seconds, n: Duration of TTS / Amount of images [DONE]
    7. Combine Concatenated Images with the Text-to-Speech [DONE]
    """

    def __init__(
        self,
        account_uuid: str,
        account_nickname: str,
        fp_profile_path: str,
        niche: str,
        language: str,
        dialect: str = "",
        character_context: str = "",
        is_for_kids: bool | None = None,
        open_browser: bool = True,
    ) -> None:
        """
        Constructor for YouTube Class.

        Args:
            account_uuid (str): The unique identifier for the YouTube account.
            account_nickname (str): The nickname for the YouTube account.
            fp_profile_path (str): Path to the firefox profile that is logged into the specificed YouTube Account.
            niche (str): The niche of the provided YouTube Channel.
            language (str): The language of the Automation.

        Returns:
            None
        """
        self._account_uuid: str = account_uuid
        self._account_nickname: str = account_nickname
        self._fp_profile_path: str = fp_profile_path
        self._niche: str = niche
        self._language: str = language
        self._dialect: str = dialect.strip()
        self._character_context: str = character_context.strip()
        self._is_for_kids: bool = get_is_for_kids() if is_for_kids is None else bool(is_for_kids)
        self.browser: webdriver.Firefox | None = None
        self._workspace_dir: str | None = None
        # Optional per-channel overrides stored on the account record
        # (script_model, asset_strategy, max_ai_assets); config.json is the fallback.
        self._account_settings: dict = next(
            (account for account in get_accounts("youtube") if account.get("id") == account_uuid),
            {},
        )

        self.images = []
        self.visual_assets = []
        self.scene_units = []
        self._stock_backups: dict[int, dict] = {}
        self._combined_meta_prompts = None
        self._cost_items = []
        self._cost_notes = []
        self._pixabay_selection_debug = []
        self._progress_callback = None
        self._manual_metadata_locked = False
        self._manual_scene_data_locked = False

        # Initialize the Firefox profile
        self.options: Options = Options()

        # Set headless state of browser
        if get_headless():
            self.options.add_argument("--headless")

        if open_browser:
            if not os.path.isdir(self._fp_profile_path):
                raise ValueError(
                    f"Firefox profile path does not exist or is not a directory: {self._fp_profile_path}"
                )

            self._assert_profile_is_available(self._fp_profile_path)

            # -no-remote / -new-instance force a fresh Firefox process bound to
            # THIS profile. Without them, launching while another Firefox is
            # open re-attaches to that running instance (and its logged-in
            # account), so account switches silently use the wrong session.
            self.options.add_argument("-no-remote")
            self.options.add_argument("-new-instance")
            self.options.add_argument("-profile")
            self.options.add_argument(self._fp_profile_path)

            # Set the service
            self.service: Service = Service(GeckoDriverManager().install())

            # Initialize the browser
            self.browser = webdriver.Firefox(
                service=self.service, options=self.options
            )

    def close(self) -> None:
        """
        Closes the Selenium browser if one is open. Safe to call multiple times.
        Call this when switching accounts so the next account opens its own
        session instead of re-using this one.

        Returns:
            None
        """
        browser = getattr(self, "browser", None)
        if browser is not None:
            try:
                browser.quit()
            except Exception:
                pass
            finally:
                self.browser = None

    def _assert_profile_is_available(self, profile_path: str) -> None:
        """
        Ensures the Firefox profile is not currently locked by another running
        Firefox process before Selenium tries to launch it.

        Args:
            profile_path (str): Firefox profile folder

        Returns:
            None
        """
        lock_path = os.path.join(profile_path, "parent.lock")
        if not os.path.exists(lock_path):
            return

        # parent.lock often lingers after an unclean exit (crash or an errored
        # Selenium session), so its mere presence does NOT mean Firefox is open.
        # A lock held by a running Firefox cannot be renamed/removed on Windows,
        # while a stale one can — use that to tell them apart and clear stale ones.
        probe_path = lock_path + ".mpv2check"
        try:
            os.replace(lock_path, probe_path)
        except OSError:
            raise RuntimeError(
                "The selected Firefox profile is currently in use. Close Firefox completely and try again."
            )
        # Rename succeeded => the lock was stale. Remove it so Firefox can launch.
        try:
            os.remove(probe_path)
        except OSError:
            pass

    @property
    def niche(self) -> str:
        """
        Getter Method for the niche.

        Returns:
            niche (str): The niche
        """
        return self._niche

    @property
    def language(self) -> str:
        """
        Getter Method for the language to use.

        Returns:
            language (str): The language
        """
        return self._language

    @property
    def dialect(self) -> str:
        """
        Getter Method for the dialect/style to use.

        Returns:
            dialect (str): The dialect/style
        """
        return self._dialect

    def set_progress_callback(self, callback) -> None:
        """
        Sets an optional callback used by interactive UIs to receive live
        generation progress updates.

        Args:
            callback: callable that accepts one dict payload

        Returns:
            None
        """
        self._progress_callback = callback if callable(callback) else None

    def _emit_progress(self, stage: str, message: str, payload: dict | None = None) -> None:
        """
        Sends one structured progress update to the registered callback.

        Args:
            stage (str): logical generation stage
            message (str): human-readable status line
            payload (dict | None): optional structured details

        Returns:
            None
        """
        if not callable(self._progress_callback):
            return

        try:
            self._progress_callback(
                {
                    "stage": str(stage or "").strip(),
                    "message": str(message or "").strip(),
                    "payload": payload or {},
                }
            )
        except Exception:
            pass

    def generate_response(
        self,
        prompt: str,
        model_name: str = None,
        instructions: str | None = None,
        max_tokens: int | None = None,
        cache_key: str | None = None,
    ) -> str:
        """
        Generates an LLM Response based on a prompt and the user-provided model.

        Args:
            prompt (str): The prompt to use in the text generation.
            model_name (str): optional model override
            instructions (str | None): static, cacheable system instructions
            max_tokens (int | None): cap on generated output tokens
            cache_key (str | None): stable prompt-cache routing key

        Returns:
            response (str): The generated AI Repsonse.
        """
        result = generate_text_result(
            prompt,
            model_name=model_name,
            instructions=instructions,
            max_tokens=max_tokens,
            cache_key=cache_key,
        )
        self._record_text_generation_cost(
            provider=result.get("provider", get_llm_provider()),
            model=result.get("model", model_name or get_configured_llm_model()),
            usage=result.get("usage", {}),
        )
        return result["text"]

    def generate_response_with_provider(
        self,
        prompt: str,
        model_name: str = None,
        provider: str | None = None,
        instructions: str | None = None,
        max_tokens: int | None = None,
        cache_key: str | None = None,
    ) -> str:
        """
        Generates an LLM response with an explicit provider override.

        Args:
            prompt (str): text-generation prompt
            model_name (str | None): optional model override
            provider (str | None): provider override
            instructions (str | None): static, cacheable system instructions
            max_tokens (int | None): cap on generated output tokens
            cache_key (str | None): stable prompt-cache routing key

        Returns:
            response (str): generated text
        """
        result = generate_text_result(
            prompt,
            model_name=model_name,
            provider=provider,
            instructions=instructions,
            max_tokens=max_tokens,
            cache_key=cache_key,
        )
        self._record_text_generation_cost(
            provider=result.get("provider", provider or get_llm_provider()),
            model=result.get("model", model_name or get_configured_llm_model()),
            usage=result.get("usage", {}),
        )
        return result["text"]

    def _llm_options(self, kind: str) -> dict:
        """
        Returns shared generation options (output-token cap + prompt-cache key)
        for a given call kind, so token controls stay consistent across calls.

        Args:
            kind (str): logical call kind (e.g. "script", "prompts", "metadata")

        Returns:
            options (dict): kwargs to splat into generate_response(...)
        """
        caps = get_llm_max_output_tokens()
        cache_enabled = get_llm_prompt_caching_enabled()
        return {
            "max_tokens": caps.get(kind) or None,
            "cache_key": f"yt-{kind}" if cache_enabled else None,
        }

    def _get_pricing_currency(self) -> str:
        """
        Returns the configured report currency.

        Returns:
            currency (str): pricing currency label
        """
        pricing = get_pricing_config()
        return str(pricing.get("currency", "USD") or "USD")

    def _get_pricing_model_config(
        self,
        category: str,
        provider: str,
        model_name: str | None = None,
    ) -> dict:
        """
        Resolves model-specific pricing config for a provider/category pair.

        Args:
            category (str): pricing section
            provider (str): provider name
            model_name (str | None): model identifier

        Returns:
            entry (dict): pricing entry or empty dict
        """
        pricing = get_pricing_config()
        category_cfg = pricing.get(category, {}) if isinstance(pricing, dict) else {}
        provider_cfg = category_cfg.get(str(provider or "").strip().lower(), {})
        models_cfg = provider_cfg.get("models", {}) if isinstance(provider_cfg, dict) else {}
        normalized_model = str(model_name or "").strip()

        if normalized_model and normalized_model in models_cfg:
            return models_cfg[normalized_model]

        lowered_lookup = {
            str(key).strip().lower(): value
            for key, value in models_cfg.items()
            if isinstance(key, str)
        }
        if normalized_model and normalized_model.lower() in lowered_lookup:
            return lowered_lookup[normalized_model.lower()]

        return models_cfg.get("default", {}) if isinstance(models_cfg, dict) else {}

    def _add_cost_note(self, note: str) -> None:
        """
        Adds a human-readable note to the current pricing report.

        Args:
            note (str): note text

        Returns:
            None
        """
        cleaned = str(note or "").strip()
        if cleaned and cleaned not in self._cost_notes:
            self._cost_notes.append(cleaned)

    def _record_cost_item(
        self,
        *,
        category: str,
        provider: str,
        model: str,
        item_type: str,
        estimated_cost: float,
        quantity: float,
        unit: str,
        details: dict | None = None,
        label: str | None = None,
    ) -> None:
        """
        Appends one normalized cost entry to the current run ledger.

        Args:
            category (str): pricing category
            provider (str): provider name
            model (str): model or service identifier
            item_type (str): granular item type
            estimated_cost (float): estimated USD amount
            quantity (float): measured quantity
            unit (str): quantity unit
            details (dict | None): extra context
            label (str | None): optional display label

        Returns:
            None
        """
        self._cost_items.append(
            {
                "category": category,
                "provider": str(provider or ""),
                "model": str(model or ""),
                "type": str(item_type or ""),
                "label": str(label or item_type or category),
                "estimated_cost": round(max(0.0, float(estimated_cost or 0.0)), 6),
                "quantity": round(max(0.0, float(quantity or 0.0)), 6),
                "unit": str(unit or ""),
                "details": details or {},
            }
        )

    def _record_text_generation_cost(
        self,
        *,
        provider: str,
        model: str,
        usage: dict | None = None,
    ) -> None:
        """
        Records the estimated cost of one text-generation call.

        Args:
            provider (str): provider name
            model (str): model name
            usage (dict | None): token usage payload

        Returns:
            None
        """
        usage = usage or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        pricing_entry = self._get_pricing_model_config(
            "text_generation",
            provider,
            model,
        )
        input_rate = float(pricing_entry.get("input_per_1m_tokens", 0.0) or 0.0)
        output_rate = float(pricing_entry.get("output_per_1m_tokens", 0.0) or 0.0)
        estimated_cost = ((input_tokens / 1_000_000) * input_rate) + (
            (output_tokens / 1_000_000) * output_rate
        )

        if str(provider or "").strip().lower() == "ollama":
            self._add_cost_note(
                "Ollama text-generation cost is estimated as zero by default. "
                "If your Ollama cloud model is billable, override pricing.models.ollama in config.json."
            )

        self._record_cost_item(
            category="text_generation",
            provider=provider,
            model=model,
            item_type="llm_call",
            label="Text generation",
            estimated_cost=estimated_cost,
            quantity=input_tokens + output_tokens,
            unit="tokens",
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_rate_per_1m": input_rate,
                "output_rate_per_1m": output_rate,
            },
        )

    def _record_image_generation_cost(self, provider: str, model: str, scene_index: int | None = None) -> None:
        """
        Records the estimated cost of one successful generated image.

        Args:
            provider (str): image provider
            model (str): model name
            scene_index (int | None): optional scene index

        Returns:
            None
        """
        pricing_entry = self._get_pricing_model_config(
            "image_generation",
            provider,
            model,
        )
        quality = get_openai_image_quality() if str(provider).lower() == "openai" else ""
        estimated_cost = 0.0
        if quality and isinstance(pricing_entry.get("qualities"), dict):
            estimated_cost = float(pricing_entry["qualities"].get(quality, 0.0) or 0.0)
        else:
            estimated_cost = float(pricing_entry.get("per_image", 0.0) or 0.0)

        self._record_cost_item(
            category="image_generation",
            provider=provider,
            model=model,
            item_type="generated_image",
            label="AI image",
            estimated_cost=estimated_cost,
            quantity=1,
            unit="image",
            details={
                "scene_index": scene_index,
                "quality": quality or None,
            },
        )

    def _record_stock_asset_cost(self, provider: str, asset_type: str, scene_index: int | None = None) -> None:
        """
        Records one sourced stock asset, typically free.

        Args:
            provider (str): stock provider
            asset_type (str): image or video
            scene_index (int | None): optional scene index

        Returns:
            None
        """
        pricing_entry = self._get_pricing_model_config(
            "image_generation",
            provider,
            "default",
        )
        estimated_cost = float(
            pricing_entry.get("per_asset", pricing_entry.get("per_image", 0.0)) or 0.0
        )
        self._record_cost_item(
            category="image_generation",
            provider=provider,
            model="stock",
            item_type=f"stock_{asset_type}",
            label=f"Stock {asset_type}",
            estimated_cost=estimated_cost,
            quantity=1,
            unit=asset_type,
            details={"scene_index": scene_index},
        )

    def _record_tts_cost(self, payload: dict) -> None:
        """
        Records TTS usage from the active speech provider.

        Args:
            payload (dict): callback payload from TTS class

        Returns:
            None
        """
        provider = str(payload.get("provider", "kitten")).strip().lower() or "kitten"
        model = str(payload.get("model", "")).strip() or "default"
        audio_seconds = max(0.0, float(payload.get("audio_seconds", 0.0) or 0.0))
        audio_minutes = audio_seconds / 60 if audio_seconds > 0 else 0.0

        pricing_entry = self._get_pricing_model_config("tts", provider, model)
        per_minute_rate = float(pricing_entry.get("per_minute_audio", 0.0) or 0.0)
        per_1k_chars_rate = float(pricing_entry.get("per_1k_characters", 0.0) or 0.0)
        characters = int(payload.get("characters", 0) or 0)

        self._record_cost_item(
            category="tts",
            provider=provider,
            model=model,
            item_type="speech_generation",
            label="Voiceover",
            estimated_cost=audio_minutes * per_minute_rate + characters / 1000 * per_1k_chars_rate,
            quantity=audio_minutes,
            unit="audio_minutes",
            details={
                "audio_seconds": round(audio_seconds, 3),
                "voice": payload.get("voice"),
                "characters": payload.get("characters"),
                "rate_per_minute_audio": per_minute_rate,
            },
        )

    def _record_stt_cost(
        self,
        provider: str,
        audio_path: str,
        model: str = "default",
        label: str = "Subtitles",
    ) -> None:
        """
        Records subtitle/STT cost from the audio duration.

        Args:
            provider (str): subtitle provider key
            audio_path (str): audio file path
            model (str): model name
            label (str): display label

        Returns:
            None
        """
        audio_clip = AudioFileClip(audio_path)
        try:
            audio_seconds = max(0.0, float(audio_clip.duration or 0.0))
        finally:
            audio_clip.close()

        pricing_entry = self._get_pricing_model_config("stt", provider, model)
        audio_minutes = audio_seconds / 60 if audio_seconds > 0 else 0.0
        per_minute_rate = float(pricing_entry.get("per_minute_audio", 0.0) or 0.0)

        self._record_cost_item(
            category="stt",
            provider=provider,
            model=model,
            item_type="subtitle_generation",
            label=label,
            estimated_cost=audio_minutes * per_minute_rate,
            quantity=audio_minutes,
            unit="audio_minutes",
            details={
                "audio_seconds": round(audio_seconds, 3),
                "rate_per_minute_audio": per_minute_rate,
            },
        )

    def get_pricing_report(self) -> dict:
        """
        Builds the current pricing report for the run.

        Returns:
            report (dict): pricing summary and itemized entries
        """
        summary: dict[str, dict] = {}
        total_estimated_cost = 0.0

        for item in self._cost_items:
            category = item["category"]
            total_estimated_cost += float(item.get("estimated_cost", 0.0) or 0.0)
            bucket = summary.setdefault(
                category,
                {"estimated_cost": 0.0, "entries": 0, "quantity": 0.0, "units": []},
            )
            bucket["estimated_cost"] += float(item.get("estimated_cost", 0.0) or 0.0)
            bucket["entries"] += 1
            bucket["quantity"] += float(item.get("quantity", 0.0) or 0.0)
            unit = str(item.get("unit", "") or "")
            if unit and unit not in bucket["units"]:
                bucket["units"].append(unit)

        normalized_summary = {
            category: {
                "estimated_cost": round(values["estimated_cost"], 6),
                "entries": values["entries"],
                "quantity": round(values["quantity"], 6),
                "units": values["units"],
            }
            for category, values in summary.items()
        }

        return {
            "currency": self._get_pricing_currency(),
            "total_estimated_cost": round(total_estimated_cost, 6),
            "items": list(self._cost_items),
            "summary": normalized_summary,
            "notes": list(self._cost_notes),
        }

    def _write_pricing_report(self) -> dict:
        """
        Persists the current run pricing report into the workspace.

        Returns:
            report (dict): pricing report
        """
        report = self.get_pricing_report()
        report["workspace_dir"] = self._get_workspace_dir()
        report["video_path"] = getattr(self, "video_path", "")
        report["subject"] = getattr(self, "subject", "")
        report["language"] = self.language
        report["dialect"] = self.dialect

        pricing_path = self._get_workspace_path("pricing.json")
        with open(pricing_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        return report

    def _to_workspace_relative_path(self, value: str) -> str:
        """
        Converts an absolute workspace file path into a workspace-relative path
        for persistence.

        Args:
            value (str): absolute or relative path

        Returns:
            path (str): workspace-relative path when possible
        """
        normalized = os.path.abspath(str(value or "").strip())
        workspace_dir = os.path.abspath(self._get_workspace_dir())
        try:
            common = os.path.commonpath([workspace_dir, normalized])
        except ValueError:
            return normalized
        if common == workspace_dir:
            return os.path.relpath(normalized, workspace_dir)
        return normalized

    def _from_workspace_relative_path(self, value: str, workspace_dir: str) -> str:
        """
        Resolves a persisted workspace-relative path back into an absolute
        filesystem path.

        Args:
            value (str): persisted path
            workspace_dir (str): workspace directory

        Returns:
            path (str): absolute file path
        """
        raw = str(value or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return os.path.abspath(os.path.join(workspace_dir, raw))

    def _build_workspace_state(self) -> dict:
        """
        Builds a persisted snapshot of the current generation state so an
        interrupted final render can be resumed later without regenerating
        prompts, assets, or audio.

        Returns:
            payload (dict): persisted workspace state
        """
        workspace_dir = os.path.abspath(self._get_workspace_dir())
        return {
            "account_uuid": self._account_uuid,
            "account_nickname": self._account_nickname,
            "workspace_dir": workspace_dir,
            "subject": getattr(self, "subject", ""),
            "script": getattr(self, "script", ""),
            "metadata": getattr(self, "metadata", {}),
            "image_prompts": list(getattr(self, "image_prompts", []) or []),
            "scene_units": list(getattr(self, "scene_units", []) or []),
            "images": [
                self._to_workspace_relative_path(path)
                for path in list(getattr(self, "images", []) or [])
                if str(path or "").strip()
            ],
            "visual_assets": [
                {
                    **asset,
                    "path": self._to_workspace_relative_path(asset.get("path", "")),
                }
                for asset in list(getattr(self, "visual_assets", []) or [])
                if isinstance(asset, dict) and str(asset.get("path", "")).strip()
            ],
            "tts_path": self._to_workspace_relative_path(getattr(self, "tts_path", "")),
            "video_path": self._to_workspace_relative_path(getattr(self, "video_path", "")),
            "language": self.language,
            "dialect": self.dialect,
            "niche": self.niche,
            "character_context": self._character_context,
            "is_for_kids": self._is_for_kids,
            "cost_items": list(getattr(self, "_cost_items", []) or []),
            "cost_notes": list(getattr(self, "_cost_notes", []) or []),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _write_workspace_state(self) -> dict:
        """
        Persists the current workspace state to disk.

        Returns:
            payload (dict): saved workspace state
        """
        payload = self._build_workspace_state()
        state_path = self._get_workspace_path("run_state.json")
        with open(state_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return payload

    def load_workspace_state(self, workspace_dir: str) -> dict:
        """
        Loads a previously saved workspace state so generation can resume from
        saved assets and voiceover output.

        Args:
            workspace_dir (str): target workspace directory

        Returns:
            payload (dict): loaded workspace payload
        """
        normalized_workspace_dir = os.path.abspath(str(workspace_dir or "").strip())
        if not normalized_workspace_dir:
            raise RuntimeError("Workspace directory is required for recovery.")

        state_path = os.path.join(normalized_workspace_dir, "run_state.json")
        if not os.path.exists(state_path):
            raise RuntimeError(
                f'No recovery state was found in "{normalized_workspace_dir}".'
            )

        with open(state_path, "r", encoding="utf-8") as file:
            payload = json.load(file) or {}

        self._workspace_dir = normalized_workspace_dir
        self.subject = str(payload.get("subject", "") or "")
        self.script = str(payload.get("script", "") or "")
        self.metadata = dict(payload.get("metadata", {}) or {})
        self.image_prompts = list(payload.get("image_prompts", []) or [])
        self.scene_units = list(payload.get("scene_units", []) or [])
        self.images = [
            self._from_workspace_relative_path(path, normalized_workspace_dir)
            for path in list(payload.get("images", []) or [])
            if str(path or "").strip()
        ]
        self.visual_assets = []
        for asset in list(payload.get("visual_assets", []) or []):
            if not isinstance(asset, dict):
                continue
            restored_asset = dict(asset)
            restored_asset["path"] = self._from_workspace_relative_path(
                restored_asset.get("path", ""),
                normalized_workspace_dir,
            )
            self.visual_assets.append(restored_asset)
        self.tts_path = self._from_workspace_relative_path(
            payload.get("tts_path", ""),
            normalized_workspace_dir,
        )
        self.video_path = self._from_workspace_relative_path(
            payload.get("video_path", ""),
            normalized_workspace_dir,
        )
        self._cost_items = list(payload.get("cost_items", []) or [])
        self._cost_notes = list(payload.get("cost_notes", []) or [])
        return payload

    def _slugify_label(self, value: str) -> str:
        """
        Converts a label into a filesystem-friendly slug.

        Args:
            value (str): raw label

        Returns:
            slug (str): safe slug
        """
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip())
        slug = slug.strip("-_")
        return slug or "youtube"

    def _get_workspace_dir(self) -> str:
        """
        Returns the per-run workspace directory for generated assets.

        Returns:
            workspace_dir (str): absolute workspace path
        """
        if self._workspace_dir is None:
            runs_root = os.path.join(ROOT_DIR, ".mp", "youtube_runs")
            os.makedirs(runs_root, exist_ok=True)
            run_name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{self._slugify_label(self._account_nickname)}_"
                f"{str(uuid4())[:8]}"
            )
            self._workspace_dir = os.path.join(runs_root, run_name)
            os.makedirs(self._workspace_dir, exist_ok=True)

        return self._workspace_dir

    def _get_workspace_path(self, filename: str) -> str:
        """
        Builds an asset path inside the current workspace directory.

        Args:
            filename (str): target filename

        Returns:
            path (str): absolute asset path
        """
        return os.path.join(self._get_workspace_dir(), filename)

    def _get_asset_strategy(self) -> str:
        """Channel's asset_strategy override, else the global config value."""
        value = str(self._account_settings.get("asset_strategy") or "").strip().lower()
        return value if value in ("mixed", "pixabay_only", "ai_only") else get_asset_strategy()

    def _get_max_ai_assets(self) -> int:
        """Channel's max_ai_assets override, else the global config value."""
        try:
            return max(0, int(self._account_settings["max_ai_assets"]))
        except (KeyError, TypeError, ValueError):
            return get_max_ai_assets()

    def _get_script_model_name(self) -> str | None:
        """Channel's script_model override, else youtube_script_model, else the active model."""
        return (
            str(self._account_settings.get("script_model") or "").strip()
            or get_youtube_script_model()
            or None
        )

    def _get_metadata_model_name(self) -> str | None:
        """
        Returns a cheaper model override for metadata generation when available.

        Returns:
            model_name (str | None): model override
        """
        configured_model = get_youtube_metadata_model().strip()
        if configured_model:
            return configured_model

        if get_llm_provider() == "openai":
            return "gpt-5-nano"

        return None

    def _get_character_context_block(self) -> str:
        """
        Returns an optional prompt block that keeps generations aligned with
        the account's persistent character and voice.

        Returns:
            block (str): prompt block or empty string
        """
        if not self._character_context:
            return ""

        return (
            f"Channel character context: {self._character_context}\n"
            "Keep the topic, script, visuals, and metadata aligned with this channel identity.\n"
        )

    def _get_locale_block(self) -> str:
        """
        Returns a prompt block describing the content language and dialect.

        Returns:
            block (str): locale instructions
        """
        dialect_line = f"Dialect/Style: {self.dialect}\n" if self.dialect else ""
        return (
            f"Language: {self.language}\n"
            f"{dialect_line}"
            "If a dialect/style is specified, write naturally in that dialect while keeping the text fluent and native.\n"
        )

    def _get_visual_locale_hint(self) -> str:
        """
        Returns a concise English hint describing the real-world cultural
        setting that the visuals should reflect.

        Returns:
            hint (str): visual locale hint or empty string
        """
        normalized_language = str(self.language or "").strip().lower()
        normalized_dialect = str(self.dialect or "").strip().lower()

        if "egyptian" in normalized_dialect:
            return (
                "Egyptian setting, Egyptian people, Cairo or Egyptian urban life, "
                "Arabic-speaking environment, culturally authentic details"
            )

        if any(term in normalized_dialect for term in ("gulf", "saudi")):
            return (
                "Gulf Arab setting, Arab people, Arabic-speaking environment, "
                "culturally authentic Gulf details"
            )

        if "levantine" in normalized_dialect:
            return (
                "Levantine Arab setting, Arab people, Arabic-speaking environment, "
                "culturally authentic Levant details"
            )

        if any(term in normalized_dialect for term in ("darija", "moroccan")):
            return (
                "Moroccan setting, Moroccan people, North African Arabic-speaking environment, "
                "culturally authentic Moroccan details"
            )

        if "arabic" in normalized_language or "arabic" in normalized_dialect:
            return (
                "Arab setting, Arab people, Arabic-speaking environment, "
                "culturally authentic local details"
            )

        return ""

    def _get_visual_locale_prompt_block(self) -> str:
        """
        Returns prompt instructions that keep visuals culturally aligned with
        the chosen audience while leaving the prompt text itself in English.

        Returns:
            block (str): prompt block or empty string
        """
        hint = self._get_visual_locale_hint()
        if not hint:
            return ""

        return (
            f"Visual locale: {hint}.\n"
            "Write the prompt in English, but keep the people, places, street details, and atmosphere aligned with that locale.\n"
        )

    def _get_target_duration_block(self) -> str:
        """
        Returns a prompt block that steers the generated script toward a target
        spoken runtime.

        Returns:
            block (str): duration instructions or empty string
        """
        target_duration = get_youtube_target_duration_seconds()
        if target_duration <= 0:
            return ""

        # Measured spoken pace: Arabic narration runs ~1.8 words/s (191 words took
        # 107 s with Edge); English TTS ~2.4. Tune if a new voice runs faster/slower.
        words_per_second = 1.8 if "arabic" in str(self.language or "").lower() else 2.4
        approx_word_budget = max(20, int(round(target_duration * words_per_second)))
        return (
            f"Target spoken runtime: about {target_duration} seconds.\n"
            f"Aim for roughly {approx_word_budget} words total when read aloud naturally.\n"
            "Keep the pacing tight and concise enough to stay around that runtime.\n"
        )

    def _template_topic_from_niche(self) -> str:
        """
        Builds a topic line from the channel niche without an LLM call.

        Returns:
            topic (str): a single-sentence topic derived from the niche
        """
        niche = re.sub(r"\s+", " ", str(self.niche or "").strip())
        if not niche:
            return ""
        # Reuse the niche as a concise, specific topic sentence.
        return niche if niche.endswith((".", "!", "?")) else f"{niche}."

    def generate_topic(self, seed_topic: str = "") -> str:
        """
        Generates a topic for the video.

        Args:
            seed_topic (str): Optional title/keyword to base the video on. When
                provided it is used as the subject directly (no LLM call), so
                the script and visuals are built around the given title.

        Returns:
            topic (str): The generated (or seeded) topic.
        """
        seed_topic = str(seed_topic or "").strip()
        if seed_topic:
            # User supplied a title/keyword — build the video around it as-is.
            completion = seed_topic
        elif get_youtube_template_topic():
            # Skip the LLM entirely: derive a topic from the niche locally.
            completion = self._template_topic_from_niche()
        else:
            completion = self.generate_response(
                f"{self._get_character_context_block()}"
                f"{self._get_locale_block()}"
                f"Please generate a specific video idea that takes about the following topic: {self.niche}. "
                "Make it exactly one sentence. Only return the topic, nothing else.",
                **self._llm_options("topic"),
            )

        if not completion:
            error("Failed to generate Topic.")

        self.subject = completion
        self._emit_progress(
            "topic",
            "Generated video topic.",
            {"subject": completion},
        )
        self._write_workspace_state()

        return completion

    def generate_script(self) -> str:
        """
        Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.

        Returns:
            script (str): The script of the video.
        """
        sentence_length = get_script_sentence_length()
        # Static rules go in `instructions` so the provider can cache this large
        # prefix across videos; only the per-video subject/locale is dynamic.
        instructions = (
            "You are a short-form video scriptwriter. Follow every rule exactly:\n"
            f"- Write the script in exactly {sentence_length} short sentences.\n"
            "- Get straight to the point; never open with filler like 'welcome to this video'.\n"
            "- No markdown, no formatting, and never use a title.\n"
            "- Do not include 'voiceover', 'narrator', or any speaker labels.\n"
            "- Write ONLY the words the narrator speaks aloud. Never describe the visuals or a scene "
            "(no lines like 'صورة المشهد:', 'في المشهد:', 'Scene:'); any visual rules in the channel "
            "context apply to the images, not to this script.\n"
            "- Never prefix a sentence with a section label (no 'Hook:', 'مفاجأة ذكية:', 'النهاية:').\n"
            "- Never mention this prompt, the script itself, or the number of sentences/paragraphs.\n"
            "- Write in the language and dialect given in the request.\n"
            "- Return only the raw script text."
        )
        prompt = (
            f"{self._get_character_context_block()}"
            f"{self._get_locale_block()}"
            f"{self._get_target_duration_block()}"
            f"Subject: {self.subject}"
        )

        completion = ""
        for attempt in range(get_llm_max_retries() + 1):
            completion = re.sub(
                r"\*",
                "",
                self.generate_response(
                    prompt,
                    model_name=self._get_script_model_name(),
                    instructions=instructions,
                    **self._llm_options("script"),
                )
                or "",
            ).strip()

            if completion and len(completion) <= 5000:
                break

            if get_verbose():
                warning(
                    "Generated script was empty or too long. "
                    f"Retrying ({attempt + 1}/{get_llm_max_retries()})..."
                )

        if not completion:
            error("The generated script is empty.")
            return

        if len(completion) > 5000:
            # Last resort after exhausting retries: truncate rather than loop.
            completion = completion[:5000].rstrip()

        completion = self._strip_script_directions(completion) or completion
        self.script = completion
        self.scene_units = []
        self._combined_meta_prompts = None
        self._emit_progress(
            "script",
            "Generated narration script.",
            {"script": completion},
        )
        self._write_workspace_state()

        return completion

    def _strip_script_directions(self, script: str) -> str:
        """
        Removes stage directions and section labels the LLM sometimes writes
        into narration, which TTS would otherwise read aloud.

        Args:
            script (str): generated script

        Returns:
            script (str): narration-only script
        """
        kept_lines = []
        for line in str(script or "").splitlines():
            if SCRIPT_DIRECTION_LINE.match(line):
                continue
            line = SCRIPT_LABEL_PREFIX.sub("", line, count=1).strip()
            if line:
                kept_lines.append(line)
        return "\n".join(kept_lines)

    def _derive_subject_from_script(self, script: str) -> str:
        """
        Derives a usable topic/subject line from a provided manual script.

        Args:
            script (str): Manual script text

        Returns:
            subject (str): Derived subject line
        """
        cleaned_script = re.sub(r"\s+", " ", script).strip()
        first_sentence = re.split(r"[.!?؟\n]+", cleaned_script)[0].strip()
        derived = first_sentence or cleaned_script
        return derived[:120].strip()

    def set_manual_script(self, script: str, subject: str = "") -> None:
        """
        Uses a caller-provided script instead of generating one with the LLM.

        Args:
            script (str): Final narration script
            subject (str): Optional topic/subject override

        Returns:
            None
        """
        cleaned_script = str(script or "").strip()
        if not cleaned_script:
            raise RuntimeError("Manual script cannot be empty.")

        cleaned_subject = str(subject or "").strip()
        self.subject = cleaned_subject or self._derive_subject_from_script(cleaned_script)
        self.script = cleaned_script
        self.scene_units = []
        self._combined_meta_prompts = None
        self._emit_progress(
            "topic",
            "Using provided video subject.",
            {"subject": self.subject},
        )
        self._emit_progress(
            "script",
            "Using provided narration script.",
            {"script": self.script},
        )
        self._write_workspace_state()

    def _normalize_scene_editor_data(
        self,
        scene_units: list[str] | None,
        image_prompts: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        """
        Aligns manual scene text and prompt edits into two non-empty lists.

        Args:
            scene_units (list[str] | None): edited scene transcript chunks
            image_prompts (list[str] | None): edited scene visual prompts

        Returns:
            aligned (tuple[list[str], list[str]]): normalized scene texts/prompts
        """
        normalized_units = [
            re.sub(r"\s+", " ", str(unit or "")).strip()
            for unit in list(scene_units or [])
            if str(unit or "").strip()
        ]
        normalized_prompts = [
            self._make_provider_friendly_image_prompt(str(prompt or ""))
            for prompt in list(image_prompts or [])
            if str(prompt or "").strip()
        ]

        if not normalized_units and normalized_prompts:
            normalized_units = [prompt for prompt in normalized_prompts]
        if not normalized_prompts and normalized_units:
            normalized_prompts = [
                self._make_provider_friendly_image_prompt(unit)
                for unit in normalized_units
            ]

        if not normalized_units:
            return [], []

        target_count = max(len(normalized_units), len(normalized_prompts))
        if len(normalized_units) < target_count:
            normalized_units.extend(normalized_units[-1:] * (target_count - len(normalized_units)))
        if len(normalized_prompts) < target_count:
            fallback_prompt = normalized_prompts[-1] if normalized_prompts else self._make_provider_friendly_image_prompt(normalized_units[-1])
            normalized_prompts.extend([fallback_prompt] * (target_count - len(normalized_prompts)))

        return normalized_units[:target_count], normalized_prompts[:target_count]

    def set_manual_metadata(self, metadata: dict | None) -> None:
        """
        Uses caller-provided metadata instead of regenerating it.

        Args:
            metadata (dict | None): title/description/tags payload

        Returns:
            None
        """
        cleaned_metadata = dict(metadata or {})
        self.metadata = cleaned_metadata
        self._manual_metadata_locked = bool(cleaned_metadata)

    def set_manual_scene_data(
        self,
        scene_units: list[str] | None,
        image_prompts: list[str] | None,
    ) -> None:
        """
        Uses caller-provided scene transcript chunks and prompts.

        Args:
            scene_units (list[str] | None): ordered scene transcript chunks
            image_prompts (list[str] | None): ordered visual prompts

        Returns:
            None
        """
        normalized_units, normalized_prompts = self._normalize_scene_editor_data(
            scene_units,
            image_prompts,
        )
        self.scene_units = normalized_units
        self.image_prompts = normalized_prompts
        self._manual_scene_data_locked = bool(normalized_units and normalized_prompts)

    def apply_workspace_edits(
        self,
        workspace_dir: str | None = None,
        *,
        subject: str | None = None,
        script: str | None = None,
        metadata: dict | None = None,
        scene_units: list[str] | None = None,
        image_prompts: list[str] | None = None,
    ) -> dict:
        """
        Persists editor changes into the saved workspace state.

        Args:
            workspace_dir (str | None): optional workspace path override
            subject (str | None): updated subject
            script (str | None): updated full script
            metadata (dict | None): updated metadata
            scene_units (list[str] | None): updated scene transcript chunks
            image_prompts (list[str] | None): updated prompts

        Returns:
            payload (dict): saved workspace payload
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        if subject is not None:
            self.subject = str(subject or "").strip()
        if script is not None:
            self.script = str(script or "").strip()
        if metadata is not None:
            self.metadata = dict(metadata or {})
        if scene_units is not None or image_prompts is not None:
            current_scene_units = scene_units if scene_units is not None else list(getattr(self, "scene_units", []) or [])
            current_prompts = image_prompts if image_prompts is not None else list(getattr(self, "image_prompts", []) or [])
            normalized_units, normalized_prompts = self._normalize_scene_editor_data(
                current_scene_units,
                current_prompts,
            )
            self.scene_units = normalized_units
            self.image_prompts = normalized_prompts

        return self._write_workspace_state()

    def _parse_json_lenient(self, completion: str):
        """
        Parses a JSON object/array from an LLM completion, tolerating code
        fences and surrounding prose.

        Args:
            completion (str): raw model output

        Returns:
            parsed: decoded JSON value, or None if nothing valid was found
        """
        cleaned = str(completion or "").replace("```json", "").replace("```", "").strip()
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    continue
        return None

    def _coerce_prompt_list(self, parsed) -> list:
        """
        Normalizes a parsed JSON value into a flat list of prompt strings.

        Args:
            parsed: decoded JSON (list, dict with "image_prompts", or nested)

        Returns:
            prompts (list): list of prompt strings (possibly empty)
        """
        if isinstance(parsed, dict) and "image_prompts" in parsed:
            parsed = parsed["image_prompts"]
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            try:
                inner = json.loads(parsed[0])
                if isinstance(inner, list):
                    parsed = inner
            except Exception:
                pass
        return parsed if isinstance(parsed, list) else []

    def _derive_local_tags_and_hashtags(self) -> tuple:
        """
        Derives tags/hashtags from the subject and script locally, avoiding an
        extra LLM call when the model omits them.

        Returns:
            (tags, hashtags) (tuple[list, list])
        """
        text = f"{self.subject} {self.script}".lower()
        words = re.findall(r"[a-z؀-ۿ][a-z0-9؀-ۿ]{2,}", text)
        stop = {
            "the", "and", "for", "with", "this", "that", "you", "your", "are",
            "was", "were", "has", "have", "but", "not", "from", "they", "their",
            "what", "when", "then", "there", "about", "into", "just", "like",
            "will", "can", "all", "out", "one", "now", "how", "why", "who",
            "while", "where", "which", "been", "more", "over", "after",
            "before", "them", "its", "his", "her", "she", "him",
        }
        ordered = []
        for word in words:
            if word in stop or word in ordered:
                continue
            ordered.append(word)
            if len(ordered) >= 8:
                break
        return ordered, ordered[:3]

    def _finalize_metadata(self, metadata: dict) -> dict:
        """
        Cleans, validates, and stores generated metadata fields.

        Args:
            metadata (dict): raw metadata mapping from the LLM

        Returns:
            metadata (dict): stored, normalized metadata
        """
        title = str(metadata.get("title", "")).strip()
        description = str(metadata.get("description", "")).strip()

        raw_hashtags = metadata.get("hashtags", [])
        if isinstance(raw_hashtags, str):
            raw_hashtags = re.split(r"[,#\n]+", raw_hashtags)
        raw_tags = metadata.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[,|\n]+", raw_tags)

        hashtags = [
            str(item).strip().replace(" ", "")
            for item in raw_hashtags
            if str(item).strip()
        ][:3]
        tags = [
            str(item).strip()
            for item in raw_tags
            if str(item).strip()
        ][:8]

        if get_youtube_local_tags() and (not tags or not hashtags):
            local_tags, local_hashtags = self._derive_local_tags_and_hashtags()
            if not tags:
                tags = local_tags
            if not hashtags:
                hashtags = local_hashtags

        if not title:
            raise RuntimeError("Metadata generation did not return a title.")

        if len(title) > 80:
            title = title[:80].rstrip()

        hashtag_line = " ".join(
            hashtag if hashtag.startswith("#") else f"#{hashtag}"
            for hashtag in hashtags
        ).strip()
        if hashtag_line:
            description = f"{description}\n\n{hashtag_line}".strip()

        self.metadata = {
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "tags": tags,
        }
        self._emit_progress(
            "metadata",
            "Generated YouTube metadata.",
            {"metadata": self.metadata},
        )
        self._write_workspace_state()
        return self.metadata

    def _finalize_image_prompts(self, raw_prompts: list, scene_units: list) -> list:
        """
        Cleans, translates (if needed), and fits raw prompts to the scene count.

        Args:
            raw_prompts (list): raw prompt strings from the LLM
            scene_units (list): ordered scene beats

        Returns:
            image_prompts (list): finalized prompts (empty if none usable)
        """
        n_prompts = len(scene_units)

        # Work on the RAW prompts first: translate before wrapping so the
        # provider-friendly wrapper (locale prefix + cinematic boilerplate) is
        # applied exactly once. Wrapping first and then re-wrapping the
        # translated output buries the real per-scene content under repeated
        # boilerplate and the 24-word cap truncates it away, collapsing every
        # scene to the same string.
        source_prompts = [
            prompt
            for prompt in raw_prompts
            if isinstance(prompt, str) and prompt.strip()
        ]

        if self._image_prompts_need_translation(source_prompts):
            source_prompts = [
                prompt
                for prompt in self._translate_image_prompts_to_english(source_prompts)
                if isinstance(prompt, str) and prompt.strip()
            ]

        image_prompts = [
            self._make_provider_friendly_image_prompt(prompt)
            for prompt in source_prompts
        ]

        if not image_prompts:
            return []

        if len(image_prompts) > n_prompts:
            image_prompts = image_prompts[:n_prompts]
        if len(image_prompts) < n_prompts:
            image_prompts.extend(image_prompts[-1:] * (n_prompts - len(image_prompts)))

        return image_prompts

    def _store_image_prompts(self, image_prompts: list, scene_units: list) -> list:
        """
        Persists finalized image prompts and emits progress.

        Args:
            image_prompts (list): finalized prompts
            scene_units (list): ordered scene beats

        Returns:
            image_prompts (list): the stored prompts
        """
        self.image_prompts = image_prompts
        self.scene_units = scene_units

        success(f"Generated {len(image_prompts)} Image Prompts.")
        self._emit_progress(
            "image_prompts",
            f"Generated {len(image_prompts)} image prompts.",
            {"image_prompts": image_prompts},
        )
        self._write_workspace_state()
        return image_prompts

    def _metadata_instructions(self) -> str:
        """Static, cacheable instruction prefix for standalone metadata."""
        return (
            "You generate YouTube Shorts SEO metadata as a single JSON object. "
            "Return ONLY valid JSON (no markdown fences) with keys: "
            '"title" (natural, clickable, under 80 characters, no markdown), '
            '"description" (2 short engaging lines), '
            '"hashtags" (array of 3 short relevant hashtags, no explanations), '
            '"tags" (array of 8 short search-friendly keywords). '
            "Align everything with the niche, audience, character context, language, and dialect. "
            "Avoid spammy clickbait or keyword stuffing. No text outside the JSON object."
        )

    def _prompts_instructions(self) -> str:
        """Static, cacheable instruction prefix for standalone image prompts."""
        return (
            "You create still-image prompts for a short narrated video; each image illustrates "
            "exactly what the narrator says at that beat. Return ONLY a JSON array of strings "
            "(no markdown, no numbering, no explanation). Each prompt: 8-18 words, English only, "
            "concrete who/what/where/doing-what, cinematic and photorealistic, depicting the SPECIFIC "
            "action/object/emotion of its beat (not a generic topic image). No readable "
            "text/captions/logos/watermarks/UI, no repeated scenes, no abstract metaphors. "
            "Output exactly one prompt per scene beat, in order."
        )

    def _combined_instructions(self) -> str:
        """Static, cacheable instruction prefix for the combined metadata+prompts call."""
        return (
            "You produce all text assets for a YouTube Short as a single JSON object. "
            "Return ONLY valid JSON (no markdown fences) with keys: "
            '"title" (natural, clickable, under 80 characters, no markdown), '
            '"description" (2 short engaging lines), '
            '"hashtags" (array of 3 short hashtags), '
            '"tags" (array of 8 short search-friendly keywords), '
            '"image_prompts" (array of still-image prompts, one per scene beat, in order). '
            "Each image prompt: 8-18 words, English only, concrete who/what/where/doing-what, "
            "cinematic and photorealistic, depicting the SPECIFIC action/object/emotion of its beat "
            "(not a generic topic image); no readable text/logos/watermarks/UI, no repeats, no "
            "abstract metaphors. Align title/description/tags with the niche, audience, character "
            "context, language, and dialect. No spammy clickbait. No text outside the JSON object."
        )

    def _generate_combined_meta_prompts(self, scene_units: list):
        """
        Produces metadata and image prompts in ONE LLM call.

        Args:
            scene_units (list): ordered scene beats

        Returns:
            (metadata, prompts) (tuple[dict | None, list]): parsed metadata and
            raw prompt list, or (None, []) if the call could not be parsed.
        """
        n_prompts = len(scene_units)
        script_block = (
            f"FULL SCRIPT (for context):\n{self.script}\n\n"
            if get_youtube_send_full_script_to_prompts()
            else ""
        )
        prompt = (
            f"{self._get_character_context_block()}"
            f"{self._get_locale_block()}"
            f"{self._get_visual_locale_prompt_block()}"
            f"Subject: {self.subject}\n\n"
            f"{script_block}"
            f"Produce exactly {n_prompts} image_prompts, one per scene beat below "
            f"(image_prompts[i] must match beat[i]).\n"
            f"Scene beats in order:\n{json.dumps(scene_units, ensure_ascii=False)}"
        )

        for attempt in range(get_llm_max_retries() + 1):
            parsed = self._parse_json_lenient(
                self.generate_response(
                    prompt,
                    instructions=self._combined_instructions(),
                    **self._llm_options("combined"),
                )
            )
            if isinstance(parsed, dict):
                prompts = self._coerce_prompt_list(parsed.get("image_prompts", []))
                if str(parsed.get("title", "")).strip() and prompts:
                    return parsed, prompts
            if get_verbose():
                warning(
                    "Combined metadata+prompts response was malformed. "
                    f"Retrying ({attempt + 1}/{get_llm_max_retries()})..."
                )
        return None, []

    def _ensure_combined_meta_prompts(self):
        """
        Runs (and caches) the combined metadata+prompts call once per script,
        so generate_metadata() and generate_prompts() share a single request.

        Returns:
            combined (dict | None): cached combined result, or None to fall back
        """
        if not get_youtube_combine_metadata_and_prompts():
            return None

        cached = getattr(self, "_combined_meta_prompts", None)
        if cached is not None:
            return cached or None  # empty dict => previously failed, don't retry

        scene_units = self._get_scene_units()
        if not scene_units:
            return None

        parsed, prompts = self._generate_combined_meta_prompts(scene_units)
        if parsed is None or not prompts:
            self._combined_meta_prompts = {}  # mark as attempted-and-failed
            return None

        parsed["_image_prompts"] = prompts
        parsed["_scene_units"] = scene_units
        self._combined_meta_prompts = parsed
        return parsed

    def generate_metadata(self) -> dict:
        """
        Generates SEO metadata for the to-be-uploaded YouTube Short.

        Returns:
            metadata (dict): The generated metadata.
        """
        combined = self._ensure_combined_meta_prompts()
        if combined is not None:
            return self._finalize_metadata(combined)

        # Standalone path (combine disabled or unavailable).
        prompt = (
            f"{self._get_character_context_block()}"
            f"{self._get_locale_block()}"
            f"Subject: {self.subject}\n"
            f"Script: {self.script}"
        )

        metadata = None
        for attempt in range(get_llm_max_retries() + 1):
            parsed = self._parse_json_lenient(
                self.generate_response(
                    prompt,
                    model_name=self._get_metadata_model_name(),
                    instructions=self._metadata_instructions(),
                    **self._llm_options("metadata"),
                )
            )
            if isinstance(parsed, dict) and str(parsed.get("title", "")).strip():
                metadata = parsed
                break
            if get_verbose():
                warning(
                    "Metadata response was not valid JSON. "
                    f"Retrying ({attempt + 1}/{get_llm_max_retries()})..."
                )

        if metadata is None:
            # Cheap fallback: two tiny calls for title + description only.
            if get_verbose():
                warning("Falling back to simpler metadata generation.")
            title = self.generate_response(
                f"{self._get_character_context_block()}{self._get_locale_block()}"
                f"Generate one YouTube Shorts title under 80 characters for this subject: {self.subject}. "
                "Return only the title.",
                model_name=self._get_metadata_model_name(),
                max_tokens=self._llm_options("topic").get("max_tokens"),
            ).strip()
            description = self.generate_response(
                f"{self._get_character_context_block()}{self._get_locale_block()}"
                f"Generate a short YouTube Shorts description for this script: {self.script}. "
                "Return only the description.",
                model_name=self._get_metadata_model_name(),
                **self._llm_options("metadata"),
            ).strip()
            metadata = {
                "title": title,
                "description": description,
                "hashtags": [],
                "tags": [],
            }

        return self._finalize_metadata(metadata)

    def generate_prompts(self) -> List[str]:
        """
        Generates AI Image Prompts based on the provided Video Script.

        Returns:
            image_prompts (List[str]): Generated List of image prompts.
        """
        scene_units = self._get_scene_units()
        if not scene_units:
            raise RuntimeError("Could not derive scene units from the current script.")

        # Reuse the combined call's prompts when available (single request).
        combined = self._ensure_combined_meta_prompts()
        if combined is not None and combined.get("_image_prompts"):
            combined_units = combined.get("_scene_units", scene_units)
            image_prompts = self._finalize_image_prompts(
                combined["_image_prompts"], combined_units
            )
            if image_prompts:
                return self._store_image_prompts(image_prompts, combined_units)

        # Standalone path.
        n_prompts = len(scene_units)
        script_block = (
            f"FULL SCRIPT (for context — understand the story before generating prompts):\n{self.script}\n\n"
            if get_youtube_send_full_script_to_prompts()
            else ""
        )
        prompt = (
            f"{self._get_character_context_block()}"
            f"{self._get_visual_locale_prompt_block()}"
            f"Subject: {self.subject}\n\n"
            f"{script_block}"
            f"Generate exactly {n_prompts} image prompts, one per scene beat below "
            f"(prompt[i] must match beat[i]).\n"
            f"Scene beats in order:\n{json.dumps(scene_units, ensure_ascii=False)}"
        )

        image_prompts = []
        for attempt in range(get_llm_max_retries() + 1):
            parsed = self._parse_json_lenient(
                self.generate_response(
                    prompt,
                    instructions=self._prompts_instructions(),
                    **self._llm_options("prompts"),
                )
            )
            image_prompts = self._finalize_image_prompts(
                self._coerce_prompt_list(parsed if parsed is not None else []),
                scene_units,
            )
            if image_prompts:
                break
            if get_verbose():
                warning(
                    "No usable image prompts returned. "
                    f"Retrying ({attempt + 1}/{get_llm_max_retries()})..."
                )

        if not image_prompts:
            raise RuntimeError("Failed to generate usable image prompts after retries.")

        return self._store_image_prompts(image_prompts, scene_units)

    def _get_target_prompt_count(self) -> int:
        """
        Derives a sensible prompt count from the script sentence count.

        Returns:
            prompt_count (int): Number of prompts to request
        """
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?؟\n]+", self.script)
            if sentence.strip()
        ]
        min_prompts = max(1, get_min_image_prompts() or DEFAULT_MIN_IMAGE_PROMPTS)
        max_prompts = max(min_prompts, get_max_image_prompts() or DEFAULT_MAX_IMAGE_PROMPTS)
        sentence_count = len(sentences) or min_prompts
        return max(min_prompts, min(max_prompts, sentence_count))

    def _split_script_into_sentences_for_visuals(self) -> list[str]:
        """
        Splits the script into sentence-like visual beats.

        Returns:
            sentences (list[str]): ordered scene candidates
        """
        raw = str(self.script or "").strip()
        sentences = [
            part.strip(" \t\r\n-")
            for part in re.split(r"(?<=[\.\!\?؟])\s+|\n+", raw)
            if part and part.strip()
        ]
        return sentences

    def _split_visual_unit_into_clauses(self, unit: str) -> list[str]:
        """
        Splits one visual unit into smaller clauses when we need more scenes.

        Args:
            unit (str): current unit

        Returns:
            clauses (list[str]): smaller ordered clauses
        """
        parts = [
            part.strip(" \t\r\n-")
            for part in re.split(r"\s*[،,:;]\s*|\s+-\s+|\s+\bthen\b\s+|\s+\bwhile\b\s+|\s+\bbut\b\s+", unit, flags=re.IGNORECASE)
            if part and part.strip()
        ]
        return parts if len(parts) > 1 else [unit.strip()]

    def _split_script_into_spoken_beats(self) -> list[str]:
        """
        Splits the script into phrase-level spoken beats so scene changes can
        track the narration more closely than plain sentence-length weighting.

        Returns:
            beats (list[str]): ordered phrase-like beats
        """
        raw_units = self._split_script_into_sentences_for_visuals()
        if not raw_units:
            return []

        beats: list[str] = []
        splitter = re.compile(
            r"\s*[،,؛;:]\s*|"
            r"\s+\b(?:then|while|but|because|so|after|before|meanwhile)\b\s+|"
            r"\s+\b(?:وبعدين|بعدها|ثم|لكن|بس|عشان|علشان|وفي نفس الوقت|وبينما)\b\s+",
            flags=re.IGNORECASE,
        )

        for unit in raw_units:
            parts = [part.strip(" \t\r\n-") for part in splitter.split(unit) if part and part.strip()]
            if parts:
                beats.extend(parts)

        return beats or raw_units

    def _merge_units_evenly(self, units: list[str], target_count: int) -> list[str]:
        """
        Merges adjacent units into evenly distributed buckets so we preserve the
        full script even when we need fewer visual scenes than raw beats.

        Args:
            units (list[str]): ordered source units
            target_count (int): desired number of merged units

        Returns:
            merged (list[str]): merged ordered units
        """
        if target_count <= 0 or len(units) <= target_count:
            return [unit for unit in units if unit.strip()]

        total_units = len(units)
        merged = []
        for index in range(target_count):
            start = round(index * total_units / target_count)
            end = round((index + 1) * total_units / target_count)
            bucket = [unit.strip() for unit in units[start:end] if unit.strip()]
            if bucket:
                separator = "، " if any(self._contains_arabic_text(unit) for unit in bucket) else ", "
                merged.append(separator.join(bucket))

        return merged or [units[0]]

    def _build_scene_units(self) -> list[str]:
        """
        Builds a low-cost ordered scene plan directly from the script by using
        phrase-like beats first and then splitting longer units into clauses
        until we reach a reasonable number of visual beats.

        Returns:
            scene_units (list[str]): ordered scene units
        """
        units = self._split_script_into_spoken_beats()
        if not units:
            fallback = re.sub(r"\s+", " ", str(self.script or "").strip())
            return [fallback] if fallback else []

        min_prompts = max(1, get_min_image_prompts() or DEFAULT_MIN_IMAGE_PROMPTS)
        max_prompts = max(min_prompts, get_max_image_prompts() or len(units))
        target_count = max(min_prompts, min(max_prompts, len(units)))

        while len(units) < target_count:
            split_index = -1
            replacement = None
            best_length = 0

            for index, unit in enumerate(units):
                clauses = self._split_visual_unit_into_clauses(unit)
                if len(clauses) <= 1:
                    continue
                if len(unit) > best_length:
                    best_length = len(unit)
                    split_index = index
                    replacement = clauses

            if split_index == -1 or not replacement:
                break

            units = units[:split_index] + replacement + units[split_index + 1 :]

        units = [unit for unit in units if unit.strip()]
        if len(units) > target_count:
            units = self._merge_units_evenly(units, target_count)
        return units[:target_count]

    def _get_scene_units(self) -> list[str]:
        """
        Returns the current ordered scene units, creating them when needed.

        Returns:
            scene_units (list[str]): ordered scene units
        """
        if not getattr(self, "scene_units", None):
            self.scene_units = self._build_scene_units()
        return self.scene_units

    def _clean_image_prompt(self, prompt: str) -> str:
        """
        Removes numbering noise from LLM-generated prompts.

        Args:
            prompt (str): Raw prompt string

        Returns:
            cleaned (str): Cleaned prompt
        """
        cleaned = prompt.strip()
        cleaned = re.sub(r"^image prompt\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\d+[\).\:-]\s*", "", cleaned)
        return cleaned.strip()

    def _strip_image_prompt_boilerplate(self, prompt: str) -> str:
        """
        Removes the cinematic wrapper and locale boilerplate that
        _make_provider_friendly_image_prompt adds, so re-wrapping a prompt (or
        wrapping one where the LLM echoed the locale instruction) does not stack
        duplicate boilerplate and starve the real scene content of its word
        budget.

        Args:
            prompt (str): prompt that may already contain wrapper boilerplate

        Returns:
            cleaned (str): prompt with known boilerplate phrases removed
        """
        cleaned = str(prompt or "")

        static_phrases = [
            "vertical cinematic still image",
            "photorealistic lighting",
            "no readable text",
            "no letters",
            "no logos",
            "no watermark",
        ]

        locale_hint = self._get_visual_locale_hint()
        locale_phrases = [
            part.strip()
            for part in locale_hint.split(",")
            if part.strip()
        ]

        for phrase in static_phrases + locale_phrases:
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        return cleaned

    def _make_provider_friendly_image_prompt(self, prompt: str) -> str:
        """
        Rewrites a raw visual prompt into a shorter, safer prompt for image
        providers that often reject long scene descriptions or readable text.

        Args:
            prompt (str): raw prompt

        Returns:
            cleaned (str): provider-friendly prompt
        """
        cleaned = self._clean_image_prompt(prompt)
        cleaned = cleaned.replace('"', "").replace("'", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")

        # Strip any wrapper/locale boilerplate the prompt may already carry
        # (echoed by the LLM, or from a previously-wrapped value) so wrapping is
        # idempotent and the boilerplate never consumes the word budget below.
        cleaned = self._strip_image_prompt_boilerplate(cleaned)

        banned_phrases = (
            "readable text",
            "text on screen",
            "caption",
            "subtitle",
            "logo",
            "watermark",
            "UI overlay",
            "user interface",
            "sign with words",
            "poster text",
        )
        for phrase in banned_phrases:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

        words = cleaned.split()
        if len(words) > 24:
            cleaned = " ".join(words[:24]).strip()

        if not cleaned:
            cleaned = "cinematic vertical scene"

        locale_hint = self._get_visual_locale_hint()
        locale_prefix = f"{locale_hint}, " if locale_hint else ""

        return (
            f"Vertical cinematic still image, {locale_prefix}{cleaned}, "
            "photorealistic lighting, no readable text, no letters, no logos, no watermark."
        )

    def _image_prompts_need_translation(self, prompts: list[str]) -> bool:
        """
        Returns whether the generated prompts still contain Arabic script and
        should be normalized into concise English prompts for image providers.

        Args:
            prompts (list[str]): generated prompts

        Returns:
            needs_translation (bool): True when translation is recommended
        """
        arabic_pattern = re.compile(r"[\u0600-\u06FF]")
        return any(arabic_pattern.search(prompt or "") for prompt in prompts)

    def _translate_image_prompts_to_english(self, prompts: list[str]) -> list[str]:
        """
        Normalizes image prompts into concise English visual prompts in one
        cheaper LLM call when the original prompts contain Arabic script.

        Args:
            prompts (list[str]): raw generated prompts

        Returns:
            translated (list[str]): translated prompts, or the originals on failure
        """
        try:
            completion = (
                str(
                    self.generate_response(
                        "Translate the following JSON array of image prompts into concise English prompts for AI image generation. "
                        "Keep the same scene meaning, make them visual and cinematic, avoid readable text/logos/UI, "
                        "and return only a JSON array of strings.\n"
                        f"{self._get_visual_locale_prompt_block()}"
                        f"{json.dumps(prompts, ensure_ascii=False)}",
                        model_name=self._get_metadata_model_name(),
                    )
                )
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            translated = json.loads(completion)
            if isinstance(translated, list):
                return [str(item).strip() for item in translated if str(item).strip()]
        except Exception:
            pass

        return prompts

    def _persist_image(self, image_bytes: bytes, provider_label: str, scene_index: int | None = None) -> str:
        """
        Writes generated image bytes to a PNG file in .mp.

        Args:
            image_bytes (bytes): Image payload
            provider_label (str): Label for logging
            scene_index (int | None): aligned scene index

        Returns:
            path (str): Absolute image path
        """
        image_index = len(self.images) + 1
        image_path = self._get_workspace_path(f"image_{image_index:02d}.png")

        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        if get_verbose():
            info(f' => Wrote image from {provider_label} to "{image_path}"')

        self.images.append(image_path)
        self.visual_assets.append({"type": "image", "path": image_path, "source": provider_label, "scene_index": scene_index})
        self._emit_progress(
            "visual_asset",
            f"Saved {provider_label} image asset for scene {int(scene_index or 0) + 1}.",
            {
                "asset_type": "image",
                "source": provider_label,
                "path": image_path,
                "scene_index": scene_index,
            },
        )
        self._write_workspace_state()
        return image_path

    def _persist_binary_asset(self, asset_bytes: bytes, filename: str, source_label: str, asset_type: str, scene_index: int | None = None) -> str:
        """
        Writes a downloaded visual asset to the current workspace and tracks it
        for the final combine step.

        Args:
            asset_bytes (bytes): Downloaded file contents
            filename (str): Output filename
            source_label (str): Asset source/provider
            asset_type (str): "image" or "video"
            scene_index (int | None): aligned scene index

        Returns:
            path (str): Absolute asset path
        """
        asset_path = self._get_workspace_path(filename)
        with open(asset_path, "wb") as asset_file:
            asset_file.write(asset_bytes)

        if asset_type == "image":
            self.images.append(asset_path)

        self.visual_assets.append({"type": asset_type, "path": asset_path, "source": source_label, "scene_index": scene_index})
        if get_verbose():
            info(f' => Wrote {asset_type} from {source_label} to "{asset_path}"')
        self._emit_progress(
            "visual_asset",
            f"Saved {source_label} {asset_type} asset for scene {int(scene_index or 0) + 1}.",
            {
                "asset_type": asset_type,
                "source": source_label,
                "path": asset_path,
                "scene_index": scene_index,
            },
        )
        self._write_workspace_state()
        return asset_path

    def _get_pixabay_cache_path(self) -> str:
        return os.path.join(ROOT_DIR, ".mp", "pixabay_cache.json")

    def _load_pixabay_cache(self) -> dict:
        cache_path = self._get_pixabay_cache_path()
        if not os.path.exists(cache_path):
            return {}

        try:
            with open(cache_path, "r", encoding="utf-8-sig") as file:
                return json.load(file) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_pixabay_cache(self, payload: dict) -> None:
        # Drop expired entries so the file (read on every search) stays small.
        now_ts = int(time.time())
        payload = {
            key: entry
            for key, entry in payload.items()
            if now_ts - int(entry.get("cached_at", 0)) < STOCK_CACHE_TTL_SECONDS
        }
        cache_path = self._get_pixabay_cache_path()
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    def _cached_stock_search(self, endpoint: str, params: dict, fetch_hits) -> list[dict]:
        """
        Runs one stock-media search through the shared 24h cache, which both
        Pixabay and Pexels ask API clients to use.

        Args:
            endpoint (str): cache namespace, e.g. "api/videos" or "pexels/videos"
            params (dict): query params that identify the search
            fetch_hits (callable): performs the live request and returns hits

        Returns:
            hits (list[dict]): result hits
        """
        normalized_params = {key: params[key] for key in sorted(params)}
        cache_key = hashlib.sha256(
            json.dumps({"endpoint": endpoint, "params": normalized_params}, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache = self._load_pixabay_cache()
        cached_entry = cache.get(cache_key)
        now_ts = int(time.time())
        if cached_entry and now_ts - int(cached_entry.get("cached_at", 0)) < STOCK_CACHE_TTL_SECONDS:
            return cached_entry.get("hits", [])

        hits = fetch_hits()
        cache[cache_key] = {"cached_at": now_ts, "hits": hits}
        self._save_pixabay_cache(cache)
        return hits

    def _search_pixabay(self, endpoint: str, params: dict) -> list[dict]:
        """
        Searches Pixabay with a simple 24h cache to respect API guidance.

        Args:
            endpoint (str): API endpoint path, e.g. "api" or "api/videos"
            params (dict): query params

        Returns:
            hits (list[dict]): result hits
        """
        api_key = get_pixabay_api_key()
        if not api_key:
            return []

        def fetch_hits() -> list[dict]:
            # (connect, read) timeouts: fail fast on a stalled request instead of
            # blocking the whole asset step for up to a minute per search.
            response = requests.get(
                f"https://pixabay.com/{endpoint}/",
                params={"key": api_key, **params},
                timeout=(8, 20),
            )
            response.raise_for_status()
            return response.json().get("hits", [])

        return self._cached_stock_search(endpoint, params, fetch_hits)

    def _search_pexels(self, asset_type: str, query: str, per_page: int) -> list[dict]:
        """
        Searches Pexels for portrait media and reshapes each result like a
        Pixabay hit, so the shared scoring and selection code ranks both.

        Args:
            asset_type (str): "video" or "image"
            query (str): search words
            per_page (int): results to request

        Returns:
            hits (list[dict]): Pixabay-shaped hits tagged with source "Pexels"
        """
        api_key = get_pexels_api_key()
        if not api_key or getattr(self, "_pexels_rate_limited", False):
            return []

        endpoint = "videos/search" if asset_type == "video" else "v1/search"
        params = {"query": query, "orientation": "portrait", "per_page": per_page}

        def fetch_hits() -> list[dict]:
            response = requests.get(
                f"https://api.pexels.com/{endpoint}",
                params=params,
                headers={"Authorization": api_key},
                timeout=(8, 20),
            )
            if response.status_code == 429:
                # Free tier is 200 requests/hour; stop asking for the rest of this run.
                self._pexels_rate_limited = True
                warning("Pexels rate limit reached; using Pixabay only for the rest of this run.")
                return []
            response.raise_for_status()
            body = response.json()
            if asset_type == "video":
                hits = [self._pexels_video_to_hit(item) for item in body.get("videos", [])]
            else:
                hits = [self._pexels_photo_to_hit(item) for item in body.get("photos", [])]
            return [hit for hit in hits if hit]

        return self._cached_stock_search(f"pexels/{endpoint}", params, fetch_hits)

    def _pexels_slug_tags(self, page_url: str) -> str:
        # Pexels has no tags field; its page slug describes the clip,
        # e.g. .../video/man-reading-newspaper-3571264/ -> "man, reading, newspaper".
        slug = str(page_url or "").rstrip("/").rsplit("/", 1)[-1]
        slug = re.sub(r"-?\d+$", "", slug)
        return ", ".join(word for word in slug.split("-") if word)

    def _pexels_video_to_hit(self, video: dict) -> dict | None:
        files = [
            item for item in video.get("video_files", []) or []
            if item.get("link") and item.get("file_type") == "video/mp4" and item.get("width") and item.get("height")
        ]
        if not files:
            return None
        # Prefer portrait files closest to 720px wide, matching Pixabay's "medium" weight.
        best = min(files, key=lambda item: (item["height"] < item["width"], abs(item["width"] - 720)))
        return {
            "id": f"pexels-{video.get('id')}",
            "source": "Pexels",
            "type": "film",
            "tags": self._pexels_slug_tags(video.get("url", "")),
            "duration": video.get("duration") or 0,
            "user": (video.get("user") or {}).get("name", ""),
            "views": 0,
            "likes": 0,
            "videos": {
                "medium": {
                    "url": best["link"],
                    "width": best["width"],
                    "height": best["height"],
                    "thumbnail": video.get("image", ""),
                }
            },
        }

    def _pexels_photo_to_hit(self, photo: dict) -> dict | None:
        src = photo.get("src") or {}
        if not src.get("large2x"):
            return None
        return {
            "id": f"pexels-{photo.get('id')}",
            "source": "Pexels",
            "type": "photo",
            "tags": str(photo.get("alt") or "") or self._pexels_slug_tags(photo.get("url", "")),
            "imageWidth": photo.get("width") or 0,
            "imageHeight": photo.get("height") or 0,
            "user": photo.get("photographer", ""),
            "views": 0,
            "likes": 0,
            "largeImageURL": src["large2x"],
            "webformatURL": src.get("medium", ""),
        }

    def _derive_stock_query_from_scene(self, scene_text: str, prompt_text: str) -> str:
        """
        Builds a short Pixabay-friendly query for one scene.

        Args:
            scene_text (str): script scene unit
            prompt_text (str): generated visual prompt for that scene

        Returns:
            query (str): short search query
        """
        preferred = self._strip_pixabay_query_boilerplate(str(prompt_text or ""))
        preferred = re.sub(r"[^A-Za-z0-9\s]", " ", preferred)
        preferred = " ".join(preferred.split()[:6]).strip()
        if preferred:
            return preferred

        fallback = re.sub(r"\s+", " ", self._strip_pixabay_query_boilerplate(str(scene_text or "")).strip())
        return fallback[:80].strip()

    def _strip_pixabay_query_boilerplate(self, value: str) -> str:
        """
        Removes prompt-style boilerplate that is useful for image models but
        harmful for stock-media search queries.

        Args:
            value (str): raw prompt or scene text

        Returns:
            cleaned (str): stock-search friendly text
        """
        cleaned = str(value or "")
        patterns = [
            r"(?i)vertical cinematic still image",
            r"(?i)photorealistic lighting",
            r"(?i)no readable text",
            r"(?i)no letters",
            r"(?i)no logos?",
            r"(?i)no watermark",
            r"(?i)cinematic",
            r"(?i)photorealistic",
            r"(?i)vertical",
            r"(?i)still image",
            r"(?i)egyptian setting",
            r"(?i)egyptian people",
            r"(?i)cairo or egyptian urban life",
            r"(?i)arabic-speaking environment",
            r"(?i)culturally authentic details",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        return cleaned

    def _format_pixabay_q_value(
        self,
        terms: list[str],
        max_chars: int = 100,
        max_words: int | None = None,
    ) -> str:
        """
        Builds a compact Pixabay q string.

        Pixabay matches every word in `q` with AND semantics, so each extra word
        shrinks the result pool. Keep stock queries to a few broad words: pass
        `max_words` to hard-cap the total word count (e.g. 2 for a broad primary
        query, 1 for the broadest fallback).

        Args:
            terms (list[str]): ordered visual search terms
            max_chars (int): maximum q length
            max_words (int | None): hard cap on total words across all terms

        Returns:
            q (str): Pixabay q string
        """
        ordered_terms: list[str] = []
        seen = set()
        for raw_term in list(terms or []):
            cleaned_term = self._clean_stock_search_query(raw_term, max_words=3)
            if not cleaned_term:
                continue
            normalized = cleaned_term.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_terms.append(cleaned_term.title())

        if not ordered_terms:
            return ""

        q_parts: list[str] = []
        current_length = 0
        used_words = 0
        for term in ordered_terms:
            term_words = len(term.split())
            if max_words is not None:
                if used_words >= max_words:
                    break
                # Trim this term so the running total never exceeds the cap.
                if used_words + term_words > max_words:
                    term = " ".join(term.split()[: max_words - used_words])
                    term_words = len(term.split())
                if not term:
                    continue
            separator_length = 1 if q_parts else 0
            next_length = current_length + separator_length + len(term)
            if next_length > max_chars:
                break
            q_parts.append(term)
            current_length = next_length
            used_words += term_words

        return "+".join(q_parts)

    def _generate_pixabay_search_queries(
        self,
        scene_units: list[str],
        prompt_texts: list[str],
    ) -> dict[int, dict]:
        """
        Uses the configured Ollama model to generate short Pixabay-ready q
        strings for all scenes in one batch, with heuristic fallback data when
        the model is unavailable.

        Args:
            scene_units (list[str]): ordered scene transcript chunks
            prompt_texts (list[str]): ordered prompt texts

        Returns:
            queries_by_scene (dict[int, dict]): primary/fallback q strings
        """
        fallback_queries: dict[int, dict] = {}
        cleaned_scene_payload = []
        self._stock_topic_query = ""
        for index, scene_text in enumerate(scene_units):
            cleaned_prompt = self._strip_pixabay_query_boilerplate(
                prompt_texts[index] if index < len(prompt_texts) else ""
            )
            cleaned_scene = self._strip_pixabay_query_boilerplate(scene_text)
            heuristic_primary = self._format_pixabay_q_value(
                [
                    self._derive_stock_query_from_scene(cleaned_scene, cleaned_prompt),
                    cleaned_scene,
                ],
                max_words=2,
            )
            heuristic_fallback = self._format_pixabay_q_value(
                [
                    cleaned_prompt,
                    cleaned_scene,
                    self.subject,
                ],
                max_words=1,
            )
            fallback_queries[index] = {
                "primary_q": heuristic_primary,
                "fallback_q": heuristic_fallback if heuristic_fallback != heuristic_primary else "",
                "source": "heuristic",
                "query_generation_note": "",
            }
            cleaned_scene_payload.append(
                {
                    "scene_index": index,
                    "scene_text": cleaned_scene,
                    "prompt_text": cleaned_prompt,
                }
            )

        if not cleaned_scene_payload:
            return fallback_queries

        # Prefer the configured LLM provider (e.g. OpenAI) for accurate queries;
        # fall back to Ollama only when explicitly disabled.
        use_main_llm = get_pixabay_query_use_main_llm()
        ollama_model = str(get_ollama_model() or "").strip()
        if not use_main_llm and not ollama_model:
            return fallback_queries

        instructions = (
            "You generate stock-footage search queries as JSON. "
            'Return ONLY a JSON object: {"topic": "...", "scenes": [...]} where scenes '
            "has one object per scene with keys scene_index, primary_q, fallback_q.\n"
            "Stock sites match every word with AND and only know what a camera "
            "filmed, so translate each idea into something physically visible:\n"
            "- primary_q: 2-3 words, a concrete subject plus its action or place "
            "(e.g. 'man looking mirror', 'woman waking bed', 'hands playing guitar').\n"
            "- fallback_q: 1-2 words, the main filmable object of that scene "
            "(e.g. 'mirror', 'guitar', 'therapist').\n"
            "- topic: 1-2 word ENGLISH NOUN for the main physical thing the whole video is about, "
            "never a verb or event (e.g. 'human brain' not 'brain swap', 'sun' not "
            "'sun disappearing', 'ocean' not 'oceans vanish').\n"
            "Never use abstract concepts (change, identity, conflict, balance, skill, "
            "memory, compatibility, time) — show the person or object instead.\n"
            "Rules: English only; no '+' or punctuation, just words separated by "
            "spaces; never use cinematic/style words (vertical, photorealistic, "
            "watermark, logo, text); never include nationality or locale words "
            "(Egyptian, Cairo, Arab)."
        )
        prompt_body = (
            f"Video subject: {self._strip_pixabay_query_boilerplate(self.subject)}\n"
            f"Scenes JSON: {json.dumps(cleaned_scene_payload, ensure_ascii=False)}"
        )
        source_label = "llm" if use_main_llm else "ollama"
        source_note = (
            "Generated via LLM stock-query planner."
            if use_main_llm
            else "Generated via Ollama stock-query planner."
        )

        last_error = ""
        for attempt in range(2):
            try:
                if use_main_llm:
                    raw = self.generate_response(
                        prompt_body,
                        model_name=self._get_metadata_model_name(),
                        instructions=instructions,
                        **self._llm_options("stock_query"),
                    )
                else:
                    raw = self.generate_response_with_provider(
                        f"{instructions}\n{prompt_body}",
                        model_name=ollama_model,
                        provider="ollama",
                    )

                parsed = self._parse_json_lenient(raw)
                if isinstance(parsed, dict):
                    self._stock_topic_query = self._format_pixabay_q_value(
                        re.split(r"\s*\+\s*", str(parsed.get("topic", "") or "").strip()),
                        max_words=2,
                    )
                    parsed = parsed.get("scenes")
                applied_queries = 0
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        try:
                            scene_index = int(item.get("scene_index", -1))
                        except (TypeError, ValueError):
                            continue
                        if scene_index not in fallback_queries:
                            continue
                        primary_q = self._format_pixabay_q_value(
                            re.split(r"\s*\+\s*", str(item.get("primary_q", "") or "").strip()),
                            max_words=3,
                        )
                        fallback_q = self._format_pixabay_q_value(
                            re.split(r"\s*\+\s*", str(item.get("fallback_q", "") or "").strip()),
                            max_words=2,
                        )
                        if primary_q:
                            fallback_queries[scene_index]["primary_q"] = primary_q
                            fallback_queries[scene_index]["source"] = source_label
                            fallback_queries[scene_index]["query_generation_note"] = source_note
                            applied_queries += 1
                        if fallback_q and fallback_q != primary_q:
                            fallback_queries[scene_index]["fallback_q"] = fallback_q

                if applied_queries > 0:
                    if not self._stock_topic_query and 0 in fallback_queries:
                        # ponytail: topic came back non-English/empty; the hook scene's main
                        # object is usually the video's subject. Good enough as a last resort.
                        self._stock_topic_query = fallback_queries[0].get("fallback_q", "")
                    return fallback_queries

                last_error = "Stock-query planner returned no usable Pixabay q values."
            except Exception as exc:
                last_error = str(exc).strip() or "Unknown stock-query generation error."

            if attempt == 0:
                time.sleep(0.4)

        note = (
            "Fell back to heuristic Pixabay q values after Ollama query generation failed twice."
            if last_error
            else "Fell back to heuristic Pixabay q values."
        )
        for payload in fallback_queries.values():
            payload["query_generation_note"] = note

        if last_error and get_verbose():
            warning(f"Pixabay q generation fell back to heuristics: {last_error}")

        return fallback_queries

    def _clean_stock_search_query(self, value: str, max_words: int = 6) -> str:
        """
        Normalizes a raw scene description into a concise Pixabay-friendly query.

        Args:
            value (str): raw input text
            max_words (int): maximum retained words

        Returns:
            query (str): cleaned short query
        """
        cleaned = str(value or "")
        cleaned = re.sub(
            r"(?i)vertical cinematic still image|photorealistic lighting|no readable text|no letters|"
            r"no logos|no watermark|cinematic|photorealistic|vertical|still image|image prompt|"
            r"egyptian setting|egyptian people|cairo or egyptian urban life|arabic speaking environment|"
            r"arabic-speaking environment|culturally authentic details",
            " ",
            cleaned,
        )
        cleaned = cleaned.replace("&", " and ")
        cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", cleaned)

        stopwords = {
            "the", "a", "an", "and", "or", "for", "with", "from", "into", "onto", "that", "this",
            "these", "those", "about", "over", "under", "through", "around", "inside", "outside",
            "what", "when", "where", "which", "who", "why", "how", "happens", "happen", "later",
            "does", "do", "did", "will", "would", "could", "should", "can", "after", "before",
            "scene", "shot", "camera", "background", "style", "visual", "cinematic", "image",
            "still", "prompt", "photo", "video", "footage", "screen", "showing", "shows",
            "there", "their", "them", "they", "your", "our", "his", "her", "its", "very",
            # Locale/meta words the LLM keeps adding ("Egyptian Street Narrator"); stock
            # libraries rarely tag them, so under AND-matching they just zero out results.
            "egyptian", "egypt", "cairo", "arab", "arabic", "arabian", "narrator", "without",
        }

        words = []
        seen = set()
        for raw_word in cleaned.split():
            word = raw_word.strip().lower()
            if len(word) < 3 or word in stopwords or word.isdigit():
                continue
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) >= max_words:
                break

        return " ".join(words).strip()

    def _split_pixabay_tags(self, value: str) -> list[str]:
        """
        Splits Pixabay's comma-separated tag string into normalized phrases.

        Args:
            value (str): Pixabay tag string

        Returns:
            tags (list[str]): normalized tags
        """
        tags = []
        for item in re.split(r"\s*,\s*", str(value or "").strip()):
            cleaned = self._clean_stock_search_query(item, max_words=3)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags

    def _get_stock_global_context(self) -> dict:
        """
        Builds broader topic context for scene-level Pixabay fallbacks.

        Returns:
            context (dict): summarized topic/search context
        """
        topic_terms = []

        def add_terms(value: str, max_words: int = 4) -> None:
            cleaned = self._clean_stock_search_query(value, max_words=max_words)
            if cleaned:
                for word in cleaned.split():
                    if word not in topic_terms:
                        topic_terms.append(word)

        # Non-English subjects/tags clean down to nothing, so lead with the
        # English topic the stock-query planner returned.
        add_terms(str(getattr(self, "_stock_topic_query", "") or "").replace("+", " "), max_words=3)
        add_terms(self.subject, max_words=5)
        for tag in self.metadata.get("tags", [])[:4]:
            add_terms(tag, max_words=3)
        for hashtag in self.metadata.get("hashtags", [])[:3]:
            add_terms(str(hashtag).replace("#", " "), max_words=3)

        return {
            "subject": self._clean_stock_search_query(self.subject, max_words=5),
            "topic_terms": topic_terms[:10],
            "topic_query": " ".join(topic_terms[:6]).strip(),
        }

    def _build_scene_query_variants(
        self,
        scene_text: str,
        prompt_text: str,
        scene_intent: dict,
        global_context: dict,
    ) -> list[dict]:
        """
        Builds prioritized search variants for one scene.

        Args:
            scene_text (str): scene narration text
            prompt_text (str): image prompt aligned to the scene
            scene_intent (dict): structured scene intent
            global_context (dict): broader video topic context

        Returns:
            variants (list[dict]): ordered search variants
        """
        variants = []

        def add_variant(variant_type: str, *parts: str) -> None:
            query = self._clean_stock_search_query(" ".join(str(part or "") for part in parts), max_words=6)
            if not query:
                return
            if any(existing["query"] == query for existing in variants):
                return
            variants.append({"type": variant_type, "query": query})

        add_variant("literal_scene", self._derive_stock_query_from_scene(scene_text, prompt_text))
        add_variant("subject_action", scene_intent.get("subject", ""), scene_intent.get("action", ""))
        add_variant("subject_setting", scene_intent.get("subject", ""), scene_intent.get("setting", "") or scene_intent.get("time", ""))
        if scene_intent.get("must_show"):
            add_variant("object_closeup", scene_intent["must_show"][0], scene_intent.get("setting", ""))
        add_variant("mood_context", scene_intent.get("mood", ""), scene_intent.get("setting", "") or scene_intent.get("subject", ""))
        add_variant("global_topic", global_context.get("topic_query", ""))
        add_variant("scene_fallback", scene_text)

        return variants[:5]

    def _build_fallback_scene_stock_intent(
        self,
        scene_index: int,
        scene_text: str,
        prompt_text: str,
        global_context: dict,
    ) -> dict:
        """
        Builds a structured scene intent without an LLM when needed.

        Args:
            scene_index (int): scene order
            scene_text (str): scene narration
            prompt_text (str): visual prompt for the scene
            global_context (dict): broader subject context

        Returns:
            intent (dict): normalized scene intent
        """
        prompt_terms = self._clean_stock_search_query(prompt_text, max_words=8).split()
        scene_terms = self._clean_stock_search_query(scene_text, max_words=8).split()
        combined_terms = prompt_terms or scene_terms or global_context.get("topic_terms", [])

        mood_words = {
            "calm", "tense", "panic", "dark", "hopeful", "happy", "sad", "dramatic",
            "luxury", "busy", "quiet", "mysterious", "urgent", "stress", "chaotic",
        }
        mood = next((word for word in combined_terms if word in mood_words), "")

        subject = " ".join(combined_terms[:3]).strip()
        action = " ".join(scene_terms[3:5] or prompt_terms[3:5]).strip()
        setting = " ".join(scene_terms[5:7] or global_context.get("topic_terms", [])[2:4]).strip()
        must_show = []
        for chunk in (subject, " ".join(combined_terms[2:5]).strip(), " ".join(combined_terms[4:7]).strip()):
            normalized = self._clean_stock_search_query(chunk, max_words=3)
            if normalized and normalized not in must_show:
                must_show.append(normalized)

        intent = {
            "scene_index": scene_index,
            "scene_text": re.sub(r"\s+", " ", str(scene_text or "").strip()),
            "prompt_text": re.sub(r"\s+", " ", str(prompt_text or "").strip()),
            "subject": self._clean_stock_search_query(subject, max_words=4),
            "action": self._clean_stock_search_query(action, max_words=3),
            "setting": self._clean_stock_search_query(setting, max_words=3),
            "time": "",
            "mood": self._clean_stock_search_query(mood, max_words=2),
            "must_show": must_show[:3],
            "must_avoid": self._get_default_stock_must_avoid_terms(),
        }
        intent["query_variants"] = self._build_scene_query_variants(
            scene_text,
            prompt_text,
            intent,
            global_context,
        )
        return intent

    def _normalize_scene_stock_intent(
        self,
        raw_intent: dict | None,
        scene_index: int,
        scene_text: str,
        prompt_text: str,
        global_context: dict,
    ) -> dict:
        """
        Normalizes one raw scene-intent object into a predictable structure.

        Args:
            raw_intent (dict | None): raw LLM payload
            scene_index (int): scene order
            scene_text (str): script scene text
            prompt_text (str): aligned image prompt
            global_context (dict): broader stock-search context

        Returns:
            intent (dict): normalized scene intent
        """
        fallback = self._build_fallback_scene_stock_intent(
            scene_index,
            scene_text,
            prompt_text,
            global_context,
        )
        if not isinstance(raw_intent, dict):
            return fallback

        intent = {
            "scene_index": scene_index,
            "scene_text": re.sub(r"\s+", " ", str(scene_text or "").strip()),
            "prompt_text": re.sub(r"\s+", " ", str(prompt_text or "").strip()),
            "subject": self._clean_stock_search_query(raw_intent.get("subject", ""), max_words=4) or fallback["subject"],
            "action": self._clean_stock_search_query(raw_intent.get("action", ""), max_words=3) or fallback["action"],
            "setting": self._clean_stock_search_query(raw_intent.get("setting", ""), max_words=3) or fallback["setting"],
            "time": self._clean_stock_search_query(raw_intent.get("time", ""), max_words=3),
            "mood": self._clean_stock_search_query(raw_intent.get("mood", ""), max_words=2) or fallback["mood"],
            "must_show": [],
            "must_avoid": self._get_default_stock_must_avoid_terms(),
        }

        raw_must_show = raw_intent.get("must_show", [])
        if not isinstance(raw_must_show, list):
            raw_must_show = [raw_must_show]
        for item in raw_must_show:
            cleaned = self._clean_stock_search_query(item, max_words=3)
            if cleaned and cleaned not in intent["must_show"]:
                intent["must_show"].append(cleaned)
        if not intent["must_show"]:
            intent["must_show"] = fallback["must_show"][:]

        raw_must_avoid = raw_intent.get("must_avoid", [])
        if not isinstance(raw_must_avoid, list):
            raw_must_avoid = [raw_must_avoid]
        for item in raw_must_avoid:
            cleaned = self._clean_stock_search_query(item, max_words=3)
            if cleaned and cleaned not in intent["must_avoid"]:
                intent["must_avoid"].append(cleaned)

        query_variants = []

        def add_query_variant(variant_type: str, query: str) -> None:
            cleaned = self._clean_stock_search_query(query, max_words=6)
            if not cleaned:
                return
            if any(existing["query"] == cleaned for existing in query_variants):
                return
            query_variants.append({"type": variant_type, "query": cleaned})

        raw_variants = raw_intent.get("query_variants", [])
        if isinstance(raw_variants, list):
            for item in raw_variants:
                if isinstance(item, dict):
                    add_query_variant(str(item.get("type", "llm_variant")).strip() or "llm_variant", item.get("query", ""))
                else:
                    add_query_variant("llm_variant", str(item))

        for variant in self._build_scene_query_variants(scene_text, prompt_text, intent, global_context):
            add_query_variant(variant["type"], variant["query"])

        intent["query_variants"] = query_variants[:5]
        return intent

    def _build_scene_stock_intents(self, scene_units: list[str], prompt_texts: list[str]) -> tuple[list[dict], dict]:
        """
        Builds structured visual intents for all scenes in one cheaper LLM call,
        with heuristic fallbacks if parsing fails.

        Args:
            scene_units (list[str]): aligned script scenes
            prompt_texts (list[str]): aligned prompts

        Returns:
            intents_and_context (tuple[list[dict], dict]): normalized scene intents and global context
        """
        # Queries first: the planner also returns the English topic the global context uses.
        ai_pixabay_queries = self._generate_pixabay_search_queries(scene_units, prompt_texts)
        global_context = self._get_stock_global_context()
        fallback_intents = [
            self._build_fallback_scene_stock_intent(
                scene_index=index,
                scene_text=scene_units[index],
                prompt_text=prompt_texts[index] if index < len(prompt_texts) else "",
                global_context=global_context,
            )
            for index in range(len(scene_units))
        ]
        for index, intent in enumerate(fallback_intents):
            generated_queries = ai_pixabay_queries.get(index, {})
            if not generated_queries:
                continue
            query_variants = []
            primary_q = str(generated_queries.get("primary_q", "") or "").strip()
            fallback_q = str(generated_queries.get("fallback_q", "") or "").strip()
            if primary_q:
                query_variants.append({"type": "ai_primary_q", "query": primary_q})
            if fallback_q and fallback_q != primary_q:
                query_variants.append({"type": "ai_fallback_q", "query": fallback_q})
            existing_queries = {
                str(variant.get("query", "")).strip()
                for variant in intent.get("query_variants", [])
                if isinstance(variant, dict)
            }
            for variant in intent.get("query_variants", []):
                if isinstance(variant, dict):
                    query_variants.append(variant)
            intent["query_variants"] = []
            seen_queries = set()
            for variant in query_variants:
                cleaned_query = self._format_pixabay_q_value(
                    re.split(r"\s*\+\s*", str(variant.get("query", "") or "").strip())
                )
                if not cleaned_query or cleaned_query in seen_queries:
                    continue
                seen_queries.add(cleaned_query)
                intent["query_variants"].append(
                    {
                        "type": str(variant.get("type", "variant") or "variant"),
                        "query": cleaned_query,
                        }
                    )
            intent["pixabay_query_source"] = generated_queries.get("source", "heuristic")
            intent["query_generation_note"] = generated_queries.get("query_generation_note", "")

        if not scene_units:
            return fallback_intents, global_context

        try:
            scene_payload = [
                {
                    "scene_index": index,
                    "scene_text": scene_units[index],
                    "prompt_text": prompt_texts[index] if index < len(prompt_texts) else "",
                }
                for index in range(len(scene_units))
            ]
            completion = (
                str(
                    self.generate_response(
                        "Return only JSON.\n"
                        f"Build Pixabay stock-search scene intents for exactly {len(scene_units)} scenes.\n"
                        "For each scene return an object with these keys:\n"
                        "subject, action, setting, time, mood as short English strings.\n"
                        "must_show and must_avoid as arrays of short English phrases.\n"
                        "query_variants as an array of up to 5 objects with keys type and query.\n"
                        "Rules:\n"
                        "- English only.\n"
                        "- Keep each query between 2 and 6 words.\n"
                        "- Prefer concrete visual search terms for stock footage.\n"
                        "- Avoid brand names, captions, logos, UI terms, and abstract essay language.\n"
                        "- In must_avoid always include culturally unsafe visuals for Arab audiences such as nudity, revealing clothes, alcohol, and smoking.\n"
                        "- Query types should prioritize literal_scene, subject_action, subject_setting, object_closeup, mood_context, global_topic.\n"
                        f"Global subject: {self.subject}\n"
                        f"Tags: {', '.join(self.metadata.get('tags', []))}\n"
                        f"Hashtags: {', '.join(self.metadata.get('hashtags', []))}\n"
                        f"Scenes JSON: {json.dumps(scene_payload, ensure_ascii=False)}",
                        model_name=self._get_metadata_model_name(),
                    )
                )
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed = json.loads(completion)
            if isinstance(parsed, list) and len(parsed) == len(scene_units):
                intents = [
                    self._normalize_scene_stock_intent(
                        parsed[index],
                        scene_index=index,
                        scene_text=scene_units[index],
                        prompt_text=prompt_texts[index] if index < len(prompt_texts) else "",
                        global_context=global_context,
                    )
                    for index in range(len(scene_units))
                ]
                for index, intent in enumerate(intents):
                    generated_queries = ai_pixabay_queries.get(index, {})
                    if not generated_queries:
                        continue
                    primary_q = str(generated_queries.get("primary_q", "") or "").strip()
                    fallback_q = str(generated_queries.get("fallback_q", "") or "").strip()
                    merged_variants = []
                    if primary_q:
                        merged_variants.append({"type": "ai_primary_q", "query": primary_q})
                    if fallback_q and fallback_q != primary_q:
                        merged_variants.append({"type": "ai_fallback_q", "query": fallback_q})
                    merged_variants.extend(intent.get("query_variants", []))
                    deduped_variants = []
                    seen_queries = set()
                    for variant in merged_variants:
                        if not isinstance(variant, dict):
                            continue
                        cleaned_query = self._format_pixabay_q_value(
                            re.split(r"\s*\+\s*", str(variant.get("query", "") or "").strip())
                        )
                        if not cleaned_query or cleaned_query in seen_queries:
                            continue
                        seen_queries.add(cleaned_query)
                        deduped_variants.append(
                            {
                                "type": str(variant.get("type", "variant") or "variant"),
                                "query": cleaned_query,
                            }
                        )
                    intent["query_variants"] = deduped_variants
                    intent["pixabay_query_source"] = generated_queries.get("source", "heuristic")
                    intent["query_generation_note"] = generated_queries.get("query_generation_note", "")
                return intents, global_context
        except Exception:
            pass

        return fallback_intents, global_context

    def _tokenize_search_text(self, value: str) -> set[str]:
        """
        Turns a free-text search phrase into normalized keyword tokens.

        Args:
            value (str): source text

        Returns:
            tokens (set[str]): normalized tokens
        """
        stopwords = {
            "the", "a", "an", "and", "or", "with", "what", "if", "into", "from", "after",
            "before", "over", "under", "city", "people", "person", "scene", "vertical",
            "cinematic", "image", "still", "lighting", "photorealistic", "setting",
        }
        words = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
        return {word for word in words if len(word) > 2 and word not in stopwords}

    def _get_pixabay_image_url(self, hit: dict) -> str | None:
        """
        Returns the preferred image URL for a Pixabay image hit.

        Args:
            hit (dict): Pixabay image result

        Returns:
            url (str | None): best downloadable URL
        """
        return hit.get("largeImageURL") or hit.get("webformatURL")

    def _score_pixabay_query_variant_priority(self, variant_type: str) -> float:
        """
        Returns a preference bonus based on how semantically close a query
        variant is to the scene.

        Args:
            variant_type (str): query variant label

        Returns:
            bonus (float): additive score bonus
        """
        priorities = {
            "ai_primary_q": 9.5,
            "ai_fallback_q": 8.5,
            "literal_scene": 8.0,
            "subject_action": 7.0,
            "subject_setting": 6.0,
            "object_closeup": 5.5,
            "mood_context": 4.5,
            "global_topic": 3.0,
            "scene_fallback": 2.5,
            "llm_variant": 5.0,
        }
        return priorities.get(str(variant_type or "").strip().lower(), 3.5)

    def _get_default_stock_must_avoid_terms(self) -> list[str]:
        """
        Returns default visual exclusions so stock assets stay suitable for
        Arab audiences.

        Returns:
            terms (list[str]): normalized avoid phrases
        """
        return [
            "nudity",
            "nude body",
            "revealing clothes",
            "short dress",
            "bikini",
            "lingerie",
            "alcohol",
            "beer",
            "wine",
            "cocktail",
            "smoking",
            "cigarette",
            "vape",
            "hookah",
        ]

    def _find_pixabay_unsafe_content_matches(self, hit: dict) -> list[str]:
        """
        Detects explicit Pixabay tags we should reject for cultural-safety
        reasons before scoring the candidate.

        Args:
            hit (dict): Pixabay result hit

        Returns:
            matches (list[str]): matched unsafe terms
        """
        unsafe_phrases = {
            "short dress",
            "mini dress",
            "revealing clothes",
            "revealing clothing",
            "revealing outfit",
            "party dress",
            "club dress",
            "night club",
            # Not unsafe, just unusable next to real footage.
            "green screen",
            "blue screen",
            "chroma key",
        }
        unsafe_tokens = {
            "nudity",
            "nude",
            "naked",
            "topless",
            "lingerie",
            "bikini",
            "swimsuit",
            "underwear",
            "bra",
            "panties",
            "cleavage",
            "sexy",
            "sensual",
            "seductive",
            "alcohol",
            "alcoholic",
            "beer",
            "wine",
            "whiskey",
            "whisky",
            "vodka",
            "champagne",
            "cocktail",
            "drunk",
            "smoking",
            "cigarette",
            "cigar",
            "vape",
            "vaping",
            "hookah",
            "shisha",
            "greenscreen",
            "chromakey",
            "chroma",
            "animation",
            "animated",
            "cartoon",
            "illustration",
            "clipart",
            "vector",
        }

        searchable_text = " ".join(
            [
                str(hit.get("tags", "")).strip().lower(),
                str(hit.get("type", "")).strip().lower(),
            ]
        ).strip()
        tag_tokens = self._tokenize_search_text(searchable_text)
        matches = [phrase for phrase in sorted(unsafe_phrases) if phrase in searchable_text]
        matches.extend(token for token in sorted(unsafe_tokens) if token in tag_tokens)
        return matches

    def _get_pixabay_selection_score(self, candidate: dict | None) -> float:
        """
        Normalizes the current Pixabay score onto a 0-100 quality scale.

        Args:
            candidate (dict | None): Pixabay candidate

        Returns:
            score (float): normalized quality score
        """
        if not isinstance(candidate, dict):
            return 0.0

        raw_score = float(candidate.get("final_score", candidate.get("base_score", 0.0)) or 0.0)
        return max(0.0, min(100.0, raw_score))

    def _pixabay_candidate_meets_quality_threshold(self, candidate: dict | None) -> bool:
        """
        Returns whether a Pixabay candidate is strong enough to keep instead of
        falling back to AI generation.

        Args:
            candidate (dict | None): Pixabay candidate

        Returns:
            meets_threshold (bool): True when the candidate clears the minimum score
        """
        return self._get_pixabay_selection_score(candidate) >= PIXBAY_MIN_SELECTION_SCORE

    def _estimate_pixabay_generic_penalty(self, tag_tokens: set[str], relevance_tokens: set[str]) -> float:
        """
        Penalizes candidates whose tags are generic and weakly tied to the scene.

        Args:
            tag_tokens (set[str]): normalized Pixabay tag tokens
            relevance_tokens (set[str]): normalized scene relevance tokens

        Returns:
            penalty (float): genericity penalty
        """
        generic_tokens = {
            "nature", "beautiful", "beauty", "travel", "outdoor", "outside", "landscape",
            "background", "summer", "sunset", "sunrise", "people", "person", "lifestyle",
            "abstract", "business", "city", "light", "color", "motion", "water", "sky",
        }
        generic_overlap = len(tag_tokens & generic_tokens)
        relevant_overlap = len(tag_tokens & relevance_tokens)
        penalty = 0.0
        if relevant_overlap == 0:
            penalty += 4.0
        if generic_overlap >= 2:
            penalty += generic_overlap * 1.5
        if len(tag_tokens) <= 2:
            penalty += 1.5
        return penalty

    def _build_pixabay_candidate(
        self,
        hit: dict,
        asset_type: str,
        query_variant: dict,
        scene_intent: dict,
        preferred_duration: float | None = None,
    ) -> dict | None:
        """
        Builds one normalized scored Pixabay candidate for a scene.

        Args:
            hit (dict): Pixabay result hit
            asset_type (str): video or image
            query_variant (dict): active query variant
            scene_intent (dict): structured scene intent
            preferred_duration (float | None): scene duration target

        Returns:
            candidate (dict | None): scored candidate or None if invalid
        """
        query = str(query_variant.get("query", "")).strip()
        if not query:
            return None

        unsafe_matches = self._find_pixabay_unsafe_content_matches(hit)
        if unsafe_matches:
            return None

        if asset_type == "video":
            if not self._is_pixabay_video_duration_usable(hit, preferred_duration):
                return None
            asset_url, asset_variant = self._choose_pixabay_video_url(hit)
            if not asset_url:
                return None
            baseline_score = self._score_pixabay_video_hit(hit, query, preferred_duration)
        else:
            asset_url = self._get_pixabay_image_url(hit)
            asset_variant = ""
            if not asset_url:
                return None
            baseline_score = self._score_pixabay_image_hit(hit, query)

        tag_tokens = self._tokenize_search_text(hit.get("tags", ""))
        query_tokens = self._tokenize_search_text(query)
        subject_tokens = self._tokenize_search_text(scene_intent.get("subject", ""))
        action_tokens = self._tokenize_search_text(scene_intent.get("action", ""))
        setting_tokens = self._tokenize_search_text(scene_intent.get("setting", ""))
        time_tokens = self._tokenize_search_text(scene_intent.get("time", ""))
        mood_tokens = self._tokenize_search_text(scene_intent.get("mood", ""))
        must_show_tokens = set()
        for item in scene_intent.get("must_show", []):
            must_show_tokens.update(self._tokenize_search_text(item))
        must_avoid_tokens = set()
        for item in scene_intent.get("must_avoid", []):
            must_avoid_tokens.update(self._tokenize_search_text(item))

        query_overlap = len(query_tokens & tag_tokens)
        subject_match = len(subject_tokens & tag_tokens)
        action_match = len(action_tokens & tag_tokens)
        setting_match = len(setting_tokens & tag_tokens)
        time_match = len(time_tokens & tag_tokens)
        mood_match = len(mood_tokens & tag_tokens)
        must_show_match = len(must_show_tokens & tag_tokens)
        must_avoid_overlap = len(must_avoid_tokens & tag_tokens)

        relevance_tokens = query_tokens | subject_tokens | action_tokens | setting_tokens | time_tokens | mood_tokens | must_show_tokens
        variant_bonus = self._score_pixabay_query_variant_priority(query_variant.get("type", ""))
        generic_penalty = self._estimate_pixabay_generic_penalty(tag_tokens, relevance_tokens)
        asset_type_bonus = 3.5 if asset_type == "video" else 1.5
        structured_match_score = (
            (subject_match * 8.5)
            + (action_match * 6.0)
            + (setting_match * 5.5)
            + (time_match * 3.0)
            + (mood_match * 2.5)
            + (must_show_match * 6.5)
            + (query_overlap * 3.0)
        )
        avoid_penalty = must_avoid_overlap * 8.0
        score = baseline_score + variant_bonus + asset_type_bonus + structured_match_score - generic_penalty - avoid_penalty

        if asset_type == "video":
            video_files = hit.get("videos") or {}
            thumbnail_url = (video_files.get("tiny") or video_files.get("medium") or {}).get("thumbnail", "")
        else:
            # CLIP downsizes to 224px, so the ~150px preview is enough and far lighter.
            thumbnail_url = hit.get("previewURL") or hit.get("webformatURL") or ""

        return {
            "candidate_key": f"{asset_type}:{hit.get('id') or asset_url}",
            "source": hit.get("source", "Pixabay"),
            "thumbnail_url": thumbnail_url,
            "asset_type": asset_type,
            "asset_url": asset_url,
            "asset_variant": asset_variant,
            "hit_id": hit.get("id"),
            "user": str(hit.get("user", "")).strip().lower(),
            "user_display": str(hit.get("user", "")).strip(),
            "top_tags": self._split_pixabay_tags(hit.get("tags", ""))[:6],
            "tag_tokens": tag_tokens,
            "best_query": query,
            "best_query_type": str(query_variant.get("type", "")).strip() or "variant",
            "matched_queries": [query],
            "matched_query_types": [str(query_variant.get("type", "")).strip() or "variant"],
            "base_score": float(score),
            "selection_score": float(max(0.0, min(100.0, score))),
            "score_breakdown": {
                "baseline": round(float(baseline_score), 3),
                "variant_bonus": round(float(variant_bonus), 3),
                "asset_type_bonus": round(float(asset_type_bonus), 3),
                "subject_match": subject_match,
                "action_match": action_match,
                "setting_match": setting_match,
                "time_match": time_match,
                "mood_match": mood_match,
                "must_show_match": must_show_match,
                "must_avoid_overlap": must_avoid_overlap,
                "query_overlap": query_overlap,
                "generic_penalty": round(float(generic_penalty), 3),
                "avoid_penalty": round(float(avoid_penalty), 3),
            },
            "duration": float(hit.get("duration") or 0.0),
            "views": int(hit.get("views", 0) or 0),
            "likes": int(hit.get("likes", 0) or 0),
            "raw_hit": hit,
        }

    def _collect_pixabay_scene_candidates(
        self,
        scene_intent: dict,
        preferred_duration: float | None = None,
        max_query_variants: int = 3,
    ) -> list[dict]:
        """
        Searches Pixabay using all query variants for one scene, merges the
        results, and keeps the strongest deduplicated candidate set.

        Args:
            scene_intent (dict): normalized scene intent
            preferred_duration (float | None): scene duration target
            max_query_variants (int): maximum search variants to try

        Returns:
            candidates (list[dict]): ranked merged candidate pool
        """
        candidates_by_key: dict[str, dict] = {}
        per_page = max(4, get_pixabay_results_per_query())
        query_variants = list(scene_intent.get("query_variants", []) or [])
        if max_query_variants and len(query_variants) > max_query_variants:
            query_variants = query_variants[:max_query_variants]
        # The video's topic (e.g. "human brain") is the last resort for scenes whose
        # own queries find nothing strong; the early break below skips it otherwise.
        topic_query = str(getattr(self, "_stock_topic_query", "") or "").strip()
        searched = {str(variant.get("query", "")).replace("+", " ").lower() for variant in query_variants}
        if topic_query and topic_query.replace("+", " ").lower() not in searched:
            query_variants.append({"type": "global_topic", "query": topic_query})

        def merge_hits(search, asset_type: str, source_label: str, query_variant: dict, query: str) -> None:
            try:
                hits = search()
            except Exception as exc:
                if get_verbose():
                    warning(f'{source_label} {asset_type} lookup failed for "{query}": {exc}')
                return

            for hit in hits:
                candidate = self._build_pixabay_candidate(
                    hit,
                    asset_type=asset_type,
                    query_variant=query_variant,
                    scene_intent=scene_intent,
                    preferred_duration=preferred_duration,
                )
                if candidate is None:
                    continue
                existing = candidates_by_key.get(candidate["candidate_key"])
                if existing is None:
                    candidates_by_key[candidate["candidate_key"]] = candidate
                    continue
                if candidate["best_query"] not in existing["matched_queries"]:
                    existing["matched_queries"].append(candidate["best_query"])
                if candidate["best_query_type"] not in existing["matched_query_types"]:
                    existing["matched_query_types"].append(candidate["best_query_type"])
                if candidate["base_score"] > existing["base_score"]:
                    existing.update(
                        {
                            "best_query": candidate["best_query"],
                            "best_query_type": candidate["best_query_type"],
                            "base_score": candidate["base_score"],
                            "score_breakdown": candidate["score_breakdown"],
                            "duration": candidate["duration"],
                            "views": candidate["views"],
                            "likes": candidate["likes"],
                        }
                    )

        for variant_index, query_variant in enumerate(query_variants):
            query = str(query_variant.get("query", "")).strip()
            if not query:
                continue

            words = query.replace("+", " ")
            params = {
                "q": words,
                "safesearch": "true",
                "per_page": per_page,
            }
            image_params = {**params, "image_type": "photo", "orientation": "vertical"}
            # "film" excludes Pixabay's animations/cartoons, which clash with real footage.
            video_params = {**params, "video_type": "film"}

            merge_hits(lambda: self._search_pixabay("api/videos", video_params), "video", "Pixabay", query_variant, query)
            merge_hits(lambda: self._search_pixabay("api", image_params), "image", "Pixabay", query_variant, query)
            merge_hits(lambda: self._search_pexels("video", words, per_page), "video", "Pexels", query_variant, query)
            merge_hits(lambda: self._search_pexels("image", words, per_page), "image", "Pexels", query_variant, query)

            ranked_candidates = sorted(
                candidates_by_key.values(),
                key=lambda item: (
                    float(item.get("base_score", 0.0)),
                    len(item.get("matched_queries", [])),
                    int(item.get("views", 0)),
                    int(item.get("likes", 0)),
                ),
                reverse=True,
            )
            if ranked_candidates:
                best_score = float(self._get_pixabay_selection_score(ranked_candidates[0]))
                strong_candidate_count = len(
                    [
                        candidate
                        for candidate in ranked_candidates[:3]
                        if self._pixabay_candidate_meets_quality_threshold(candidate)
                    ]
                )
                if best_score >= 78.0 or (variant_index == 0 and strong_candidate_count >= 2):
                    break

        self._apply_clip_rerank(list(candidates_by_key.values()), scene_intent)

        return sorted(
            candidates_by_key.values(),
            key=lambda item: (
                float(item.get("base_score", 0.0)),
                len(item.get("matched_queries", [])),
                int(item.get("views", 0)),
                int(item.get("likes", 0)),
            ),
            reverse=True,
        )

    def _get_clip_model(self):
        """
        Lazily loads the CLIP model shared by every YouTube instance. Returns
        None when open_clip is not installed, which disables reranking.
        """
        cached = getattr(YouTube, "_clip_bundle", None)
        if cached is not None:
            return cached or None

        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
            model.eval()
            YouTube._clip_bundle = (model, preprocess, open_clip.get_tokenizer(CLIP_MODEL[0]), torch)
        except Exception as exc:
            warning(f"CLIP rerank disabled (pip install open_clip_torch to enable): {exc}")
            YouTube._clip_bundle = False
        return YouTube._clip_bundle or None

    def _apply_clip_rerank(self, candidates: list[dict], scene_intent: dict) -> None:
        """
        Adds a visual-match bonus to the top candidates by comparing their
        thumbnails with the scene description. Tag overlap alone misjudges
        clips because stock sites only attach 3-6 tags each.

        Args:
            candidates (list[dict]): scene candidates, updated in place
            scene_intent (dict): normalized scene intent (English fields)
        """
        scene_description = " ".join(
            str(scene_intent.get(key, "") or "").strip()
            for key in ("subject", "action", "setting", "time", "mood")
        ).strip()
        if not scene_description:
            variants = scene_intent.get("query_variants") or [{}]
            scene_description = str(variants[0].get("query", "")).replace("+", " ")
        # Every candidate is checked: an unchecked clip would keep its tag score
        # and beat checked ones that took a mismatch penalty.
        top = sorted(candidates, key=lambda item: float(item.get("base_score", 0.0)), reverse=True)
        top = top[:CLIP_RERANK_TOP_N]
        if not scene_description or not top:
            return

        bundle = self._get_clip_model()
        if bundle is None:
            return
        model, preprocess, tokenizer, torch = bundle
        from PIL import ImageStat

        def load_thumbnail(url: str):
            try:
                return Image.open(io.BytesIO(self._download_url_bytes(url))).convert("RGB")
            except Exception:
                return None

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=16) as pool:
            thumbnails = list(pool.map(load_thumbnail, [item.get("thumbnail_url") or "" for item in top]))
        scored = [(item, image) for item, image in zip(top, thumbnails) if image is not None]
        if not scored:
            return

        # A clip matches if it shows the scene OR the video's overall topic
        # (e.g. a brain shot suits any scene of a brain-swap video).
        texts = [f"a photo of {scene_description}"]
        topic = str(getattr(self, "_stock_topic_query", "") or "").replace("+", " ").strip()
        if topic:
            texts.append(f"a photo of {topic}")
        with torch.no_grad():
            image_features = model.encode_image(torch.stack([preprocess(image) for _, image in scored]))
            text_features = model.encode_text(tokenizer(texts))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity_rows = (image_features @ text_features.T).tolist()

        bonuses = []
        for (item, image), row in zip(scored, similarity_rows):
            scene_similarity = row[0]
            topic_similarity = row[1] if len(row) > 1 else 0.0
            similarity = max(scene_similarity, topic_similarity - CLIP_TOPIC_HANDICAP)
            brightness = ImageStat.Stat(image.convert("L")).mean[0]
            dark_penalty = DARK_THUMBNAIL_PENALTY if brightness < DARK_THUMBNAIL_LUMA else 0.0
            bonus = (similarity - CLIP_SIMILARITY_BASELINE) * CLIP_SIMILARITY_WEIGHT - dark_penalty
            bonuses.append(bonus)
            item["base_score"] = float(item.get("base_score", 0.0)) + bonus
            item["selection_score"] = float(max(0.0, min(100.0, item["base_score"])))
            item.setdefault("score_breakdown", {}).update(
                {
                    "clip_similarity": round(scene_similarity, 4),
                    "clip_topic_similarity": round(topic_similarity, 4),
                    "clip_bonus": round(bonus, 3),
                    "brightness": round(brightness, 1),
                    "dark_penalty": dark_penalty,
                }
            )

        # Candidates CLIP could not see (no/failed thumbnail, or beyond TOP_N) get
        # the worst observed bonus, so being unchecked is never an advantage.
        scored_ids = {id(item) for item, _ in scored}
        worst_bonus = min(bonuses)
        for item in candidates:
            if id(item) not in scored_ids:
                item["base_score"] = float(item.get("base_score", 0.0)) + worst_bonus
                item["selection_score"] = float(max(0.0, min(100.0, item["base_score"])))
                item.setdefault("score_breakdown", {})["clip_bonus"] = round(worst_bonus, 3)

    def _calculate_pixabay_diversity_penalty(self, candidate: dict, selected_candidates: list[dict]) -> tuple[float, dict]:
        """
        Penalizes near-duplicate Pixabay choices across scenes.

        Args:
            candidate (dict): candidate under consideration
            selected_candidates (list[dict]): already selected scene candidates

        Returns:
            penalty_and_breakdown (tuple[float, dict]): diversity penalty and explanation
        """
        same_uploader_count = 0
        shared_tag_penalty = 0.0
        dominant_tag_penalty = 0.0
        repeated_query_penalty = 0.0

        candidate_tags = set(candidate.get("tag_tokens", set()))
        candidate_dominant_tags = candidate.get("top_tags", [])[:3]

        for selected in selected_candidates:
            if candidate.get("user") and candidate.get("user") == selected.get("user"):
                same_uploader_count += 1

            shared_tags = len(candidate_tags & set(selected.get("tag_tokens", set())))
            if shared_tags >= 2:
                shared_tag_penalty += min(9.0, shared_tags * 2.2)

            selected_dominant_tags = selected.get("top_tags", [])[:3]
            dominant_tag_penalty += sum(
                2.5
                for tag in candidate_dominant_tags
                if tag and tag in selected_dominant_tags
            )

            if candidate.get("best_query") and candidate.get("best_query") == selected.get("best_query"):
                repeated_query_penalty += 1.5

        uploader_penalty = same_uploader_count * 6.0
        total_penalty = uploader_penalty + shared_tag_penalty + dominant_tag_penalty + repeated_query_penalty
        return total_penalty, {
            "same_uploader_count": same_uploader_count,
            "uploader_penalty": round(float(uploader_penalty), 3),
            "shared_tag_penalty": round(float(shared_tag_penalty), 3),
            "dominant_tag_penalty": round(float(dominant_tag_penalty), 3),
            "repeated_query_penalty": round(float(repeated_query_penalty), 3),
        }

    def _select_pixabay_candidates_across_scenes(self, scene_plans: list[dict]) -> dict[int, dict]:
        """
        Selects one Pixabay candidate per scene after all candidate pools are
        known, so we can penalize repeated visual patterns across the run.

        Args:
            scene_plans (list[dict]): scene plans with candidate pools

        Returns:
            selected (dict[int, dict]): selected candidate per scene index
        """
        selected: dict[int, dict] = {}
        selected_candidates: list[dict] = []
        used_candidate_keys: set[str] = set()

        scene_order = sorted(
            range(len(scene_plans)),
            key=lambda index: (
                len(scene_plans[index].get("candidates", [])),
                -float(scene_plans[index].get("candidates", [{}])[0].get("base_score", -9999.0))
                if scene_plans[index].get("candidates")
                else 0.0,
                index,
            ),
        )

        for scene_index in scene_order:
            plan = scene_plans[scene_index]
            best_choice = None

            for candidate in plan.get("candidates", []):
                if candidate["candidate_key"] in used_candidate_keys:
                    continue

                diversity_penalty, diversity_breakdown = self._calculate_pixabay_diversity_penalty(
                    candidate,
                    selected_candidates,
                )
                final_score = float(candidate.get("base_score", 0.0)) - diversity_penalty
                if best_choice is None or final_score > best_choice["final_score"]:
                    best_choice = {
                        **candidate,
                        "final_score": final_score,
                        "diversity_penalty": diversity_penalty,
                        "diversity_breakdown": diversity_breakdown,
                    }

            if best_choice is None:
                continue

            best_choice["selection_score"] = self._get_pixabay_selection_score(best_choice)
            selected[scene_index] = best_choice
            used_candidate_keys.add(best_choice["candidate_key"])
            selected_candidates.append(best_choice)

        return selected

    def _summarize_pixabay_candidate_for_debug(self, candidate: dict, selected: bool = False) -> dict:
        """
        Converts an internal Pixabay candidate into a JSON-safe debug summary.

        Args:
            candidate (dict): candidate payload
            selected (bool): whether this candidate won the scene

        Returns:
            summary (dict): JSON-safe summary
        """
        return {
            "selected": selected,
            "source": candidate.get("source", "Pixabay"),
            "asset_type": candidate.get("asset_type"),
            "asset_id": candidate.get("hit_id"),
            "asset_url": candidate.get("asset_url"),
            "asset_variant": candidate.get("asset_variant"),
            "user": candidate.get("user_display"),
            "tags": candidate.get("top_tags", []),
            "best_query": candidate.get("best_query"),
            "best_query_type": candidate.get("best_query_type"),
            "matched_queries": candidate.get("matched_queries", [])[:5],
            "matched_query_types": candidate.get("matched_query_types", [])[:5],
            "base_score": round(float(candidate.get("base_score", 0.0)), 3),
            "final_score": round(float(candidate.get("final_score", candidate.get("base_score", 0.0))), 3),
            "selection_score": round(float(self._get_pixabay_selection_score(candidate)), 3),
            "quality_threshold": round(float(PIXBAY_MIN_SELECTION_SCORE), 3),
            "meets_quality_threshold": self._pixabay_candidate_meets_quality_threshold(candidate),
            "diversity_penalty": round(float(candidate.get("diversity_penalty", 0.0)), 3),
            "score_breakdown": candidate.get("score_breakdown", {}),
            "diversity_breakdown": candidate.get("diversity_breakdown", {}),
            "duration": round(float(candidate.get("duration", 0.0)), 3),
            "views": int(candidate.get("views", 0) or 0),
            "likes": int(candidate.get("likes", 0) or 0),
        }

    def _write_pixabay_debug_report(self, payload: dict) -> None:
        """
        Writes a detailed Pixabay selection report into the run workspace.

        Args:
            payload (dict): debug payload

        Returns:
            None
        """
        self._pixabay_selection_debug = payload.get("scenes", [])
        debug_path = self._get_workspace_path("pixabay_selection_debug.json")
        with open(debug_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def _plan_pixabay_scene_assets(
        self,
        scene_units: list[str],
        prompt_texts: list[str],
        preferred_scene_durations: list[float],
        scene_indices: list[int] | None = None,
    ) -> dict[int, dict]:
        """
        Plans, selects, downloads, and tracks the best Pixabay assets for all
        scenes in one coordinated pass.

        Args:
            scene_units (list[str]): ordered scene texts
            prompt_texts (list[str]): aligned image prompts
            preferred_scene_durations (list[float]): estimated scene durations
            scene_indices (list[int] | None): optional original scene indices

        Returns:
            assets_by_scene (dict[int, dict]): downloaded asset descriptors by scene
        """
        assets_by_scene: dict[int, dict] = {}
        actual_scene_indices = (
            [int(index) for index in scene_indices]
            if scene_indices is not None
            else list(range(len(scene_units)))
        )

        if len(actual_scene_indices) != len(scene_units):
            raise ValueError("scene_indices must align with scene_units.")

        scene_intents, global_context = self._build_scene_stock_intents(scene_units, prompt_texts)
        scene_plans = []
        total_scenes = len(scene_units)

        for local_scene_index, scene_text in enumerate(scene_units):
            scene_index = actual_scene_indices[local_scene_index]
            self._emit_progress(
                "assets_search",
                f"Searching stock footage for scene {local_scene_index + 1}/{total_scenes}...",
            )
            prompt_text = prompt_texts[local_scene_index] if local_scene_index < len(prompt_texts) else ""
            scene_intent = scene_intents[local_scene_index] if local_scene_index < len(scene_intents) else self._build_fallback_scene_stock_intent(
                scene_index,
                scene_text,
                prompt_text,
                global_context,
            )
            preferred_duration = preferred_scene_durations[local_scene_index] if local_scene_index < len(preferred_scene_durations) else None
            candidates = self._collect_pixabay_scene_candidates(scene_intent, preferred_duration=preferred_duration)
            scene_plans.append(
                {
                    "local_scene_index": local_scene_index,
                    "scene_index": scene_index,
                    "scene_text": scene_text,
                    "prompt_text": prompt_text,
                    "preferred_duration": preferred_duration,
                    "scene_intent": scene_intent,
                    "candidates": candidates,
                }
            )

        selected_by_scene = self._select_pixabay_candidates_across_scenes(scene_plans)
        debug_scenes = []
        # With no AI fallback, a weaker stock clip beats repeating the previous scene's clip.
        accept_below_threshold = self._get_asset_strategy() == "pixabay_only"

        for plan in scene_plans:
            scene_index = plan["scene_index"]
            local_scene_index = plan["local_scene_index"]
            selected_candidate = selected_by_scene.get(local_scene_index)
            approved_candidate = (
                selected_candidate
                if accept_below_threshold or self._pixabay_candidate_meets_quality_threshold(selected_candidate)
                else None
            )
            selected_summary = None
            local_asset_path = None
            fallback_reason = ""

            if selected_candidate is not None and approved_candidate is None:
                fallback_reason = (
                    f"Pixabay score {self._get_pixabay_selection_score(selected_candidate):.1f} "
                    f"is below the minimum {PIXBAY_MIN_SELECTION_SCORE:.1f}, so this scene will fall back to AI."
                )
                selected_summary = self._summarize_pixabay_candidate_for_debug(selected_candidate, selected=False)
                selected_summary["rejected_for_ai_fallback"] = True
                selected_summary["rejection_reason"] = fallback_reason
                # Kept (not downloaded) as the scene's backup in case the AI budget runs out.
                self._stock_backups[scene_index] = selected_candidate

            if approved_candidate is not None:
                try:
                    asset_record = self._download_stock_candidate(approved_candidate, scene_index)
                    if asset_record:
                        assets_by_scene[scene_index] = asset_record
                        local_asset_path = asset_record.get("path")
                        selected_summary = self._summarize_pixabay_candidate_for_debug(approved_candidate, selected=True)
                except Exception as exc:
                    if get_verbose():
                        warning(f'Failed to download selected Pixabay asset for scene {scene_index + 1}: {exc}')
                    fallback_reason = f"Pixabay download failed, so this scene will fall back to AI: {exc}"

            top_candidates = []
            for candidate in plan.get("candidates", [])[:5]:
                is_selected = bool(approved_candidate and candidate["candidate_key"] == approved_candidate["candidate_key"])
                candidate_for_debug = approved_candidate if is_selected else candidate
                top_candidates.append(self._summarize_pixabay_candidate_for_debug(candidate_for_debug, selected=is_selected))

            debug_scenes.append(
                {
                    "scene_index": scene_index,
                    "scene_text": plan["scene_text"],
                    "prompt_text": plan["prompt_text"],
                    "preferred_duration": round(float(plan["preferred_duration"] or 0.0), 3),
                    "intent": {
                        "subject": plan["scene_intent"].get("subject", ""),
                        "action": plan["scene_intent"].get("action", ""),
                        "setting": plan["scene_intent"].get("setting", ""),
                        "time": plan["scene_intent"].get("time", ""),
                        "mood": plan["scene_intent"].get("mood", ""),
                        "must_show": plan["scene_intent"].get("must_show", []),
                        "must_avoid": plan["scene_intent"].get("must_avoid", []),
                        "pixabay_query_source": plan["scene_intent"].get("pixabay_query_source", "heuristic"),
                    },
                    "query_variants": plan["scene_intent"].get("query_variants", []),
                    "candidate_count": len(plan.get("candidates", [])),
                    "quality_threshold": round(float(PIXBAY_MIN_SELECTION_SCORE), 3),
                    "requires_ai_fallback": scene_index not in assets_by_scene,
                    "fallback_reason": fallback_reason,
                    "selected_asset": selected_summary,
                    "local_asset_path": local_asset_path,
                    "top_candidates": top_candidates,
                }
            )

        self._write_pixabay_debug_report(
            {
                "subject": self.subject,
                "script": self.script,
                "global_context": global_context,
                "scenes": debug_scenes,
            }
        )
        return assets_by_scene

    def _download_stock_candidate(self, candidate: dict, scene_index: int) -> dict | None:
        """
        Downloads one scored stock candidate and records it as the scene's asset.

        Args:
            candidate (dict): scored Pixabay/Pexels candidate
            scene_index (int): aligned scene order

        Returns:
            asset (dict | None): stored asset record, or None on an empty download
        """
        asset_bytes = self._download_url_bytes(candidate["asset_url"])
        if not asset_bytes:
            return None

        if candidate["asset_type"] == "video":
            filename = (
                f"stock_video_{len(self.visual_assets)+1:02d}_"
                f"{candidate.get('hit_id') or 'pixabay'}_"
                f"{candidate.get('asset_variant') or 'clip'}.mp4"
            )
        else:
            filename = f"stock_image_{len(self.visual_assets)+1:02d}_{candidate.get('hit_id') or 'pixabay'}.jpg"

        source_label = candidate.get("source", "Pixabay")
        path = self._persist_binary_asset(
            asset_bytes,
            filename,
            source_label,
            candidate["asset_type"],
            scene_index=scene_index,
        )
        self._record_stock_asset_cost(source_label.lower(), candidate["asset_type"], scene_index=scene_index)
        if self.visual_assets and self.visual_assets[-1].get("path") == path:
            self.visual_assets[-1].update(
                {
                    "pixabay_hit_id": candidate.get("hit_id"),
                    "pixabay_query": candidate.get("best_query"),
                    "pixabay_query_type": candidate.get("best_query_type"),
                    "pixabay_tags": candidate.get("top_tags", []),
                }
            )
            return self.visual_assets[-1]
        return {
            "type": candidate["asset_type"],
            "path": path,
            "source": source_label,
            "scene_index": scene_index,
        }

    def _get_existing_visual_asset_scene_indices(self, target_count: int) -> set[int]:
        """
        Returns the set of scene indices that already have a tracked visual asset.

        Args:
            target_count (int): maximum valid scene count

        Returns:
            scene_indices (set[int]): completed scene indices
        """
        scene_indices: set[int] = set()

        for asset in list(getattr(self, "visual_assets", []) or []):
            if not isinstance(asset, dict):
                continue
            scene_index = asset.get("scene_index")
            try:
                normalized_index = int(scene_index)
            except (TypeError, ValueError):
                continue

            if 0 <= normalized_index < int(target_count):
                scene_indices.add(normalized_index)

        return scene_indices

    def _estimate_scene_durations_for_asset_search(self) -> list[float]:
        """
        Estimates scene durations before TTS is rendered so we can prefer
        stock videos whose native duration is close to the intended scene.

        Returns:
            durations (list[float]): estimated scene durations
        """
        target_total_duration = max(5, get_youtube_target_duration_seconds())
        return self._get_scene_durations(float(target_total_duration))

    def _score_pixabay_video_hit(self, hit: dict, query: str, preferred_duration: float | None = None) -> float:
        """
        Scores a Pixabay video hit for suitability.

        Args:
            hit (dict): Pixabay hit
            query (str): scene query
            preferred_duration (float | None): desired scene duration

        Returns:
            score (float): higher is better
        """
        query_tokens = self._tokenize_search_text(query)
        tag_tokens = self._tokenize_search_text(hit.get("tags", ""))
        overlap = len(query_tokens & tag_tokens)

        duration = float(hit.get("duration") or 0)
        duration_score = 0.0
        if preferred_duration and duration > 0:
            duration_score = max(0.0, 8.0 - abs(duration - preferred_duration))
            if duration > max(18.0, preferred_duration * 3.5):
                duration_score -= 10.0

        views_score = min(8.0, float(hit.get("views", 0)) / 150000.0)
        likes_score = min(6.0, float(hit.get("likes", 0)) / 400.0)
        quality_bonus = 3.0 if not hit.get("isLowQuality", False) else -4.0
        ai_penalty = -3.0 if hit.get("isAiGenerated", False) else 0.0
        film_bonus = 2.0 if str(hit.get("type", "")).lower() == "film" else 0.0
        first_file = next(iter((hit.get("videos") or {}).values()), {}) or {}
        portrait_bonus = (
            PORTRAIT_VIDEO_BONUS
            if (first_file.get("height") or 0) > (first_file.get("width") or 0)
            else 0.0
        )

        return (
            overlap * 10.0 + duration_score + views_score + likes_score
            + quality_bonus + ai_penalty + film_bonus + portrait_bonus
        )

    def _is_pixabay_video_duration_usable(self, hit: dict, preferred_duration: float | None) -> bool:
        """
        Returns whether a Pixabay video duration is reasonable for the target
        scene duration.

        Args:
            hit (dict): Pixabay video hit
            preferred_duration (float | None): desired scene duration

        Returns:
            usable (bool): True if the duration is acceptable
        """
        duration = float(hit.get("duration") or 0)
        if duration <= 0:
            return False
        if preferred_duration is None:
            return duration <= 20.0
        return duration <= max(18.0, preferred_duration * 3.5)

    def _score_pixabay_image_hit(self, hit: dict, query: str) -> float:
        """
        Scores a Pixabay image hit for suitability.

        Args:
            hit (dict): Pixabay hit
            query (str): scene query

        Returns:
            score (float): higher is better
        """
        query_tokens = self._tokenize_search_text(query)
        tag_tokens = self._tokenize_search_text(hit.get("tags", ""))
        overlap = len(query_tokens & tag_tokens)

        width = float(hit.get("imageWidth") or 0)
        height = float(hit.get("imageHeight") or 0)
        orientation_score = 0.0
        if width > 0 and height > 0:
            ratio = width / height
            orientation_score = max(0.0, 4.0 - abs(ratio - 0.5625) * 6.0)

        views_score = min(6.0, float(hit.get("views", 0)) / 200000.0)
        likes_score = min(5.0, float(hit.get("likes", 0)) / 250.0)
        quality_bonus = 2.0 if not hit.get("isLowQuality", False) else -3.0
        ai_penalty = -2.0 if hit.get("isAiGenerated", False) else 0.0

        return overlap * 10.0 + orientation_score + views_score + likes_score + quality_bonus + ai_penalty

    def _download_url_bytes(self, url: str) -> bytes | None:
        if not url:
            return None
        # (connect, read) timeouts so a stalled download can't freeze the run.
        # Shared session reuses TLS connections: dozens of CDN thumbnails per scene.
        response = HTTP_SESSION.get(url, timeout=(8, 45))
        response.raise_for_status()
        return response.content

    def _choose_pixabay_video_url(self, hit: dict) -> tuple[str | None, str]:
        videos = hit.get("videos", {}) or {}
        for key in ("medium", "large", "small", "tiny"):
            item = videos.get(key) or {}
            url = item.get("url")
            if url:
                return url, key
        return None, ""

    def _generate_visual_assets(self) -> int:
        """
        Creates visual assets based on the configured sourcing strategy.

        Returns:
            count (int): total generated/downloaded assets
        """
        strategy = self._get_asset_strategy()
        scene_units = self._get_scene_units()
        target_count = len(scene_units)
        ai_used = 0
        estimated_scene_durations = self._estimate_scene_durations_for_asset_search()
        self._pixabay_selection_debug = []
        self._stock_backups = {}
        pixabay_assets_by_scene: dict[int, dict] = {}

        if strategy in ("mixed", "pixabay_only") and scene_units:
            prompt_texts = [
                self.image_prompts[index] if index < len(self.image_prompts) else self.image_prompts[-1]
                for index in range(len(scene_units))
            ]
            pixabay_assets_by_scene = self._plan_pixabay_scene_assets(
                scene_units,
                prompt_texts,
                estimated_scene_durations,
            )

        max_ai_assets = self._get_max_ai_assets() if strategy in ("mixed", "ai_only") else 0

        # Spend the AI budget where stock is weakest: scenes with no candidate
        # first, then the lowest-scoring rejected ones.
        missing = [index for index in range(target_count) if index not in pixabay_assets_by_scene]
        missing.sort(key=lambda index: self._get_pixabay_selection_score(self._stock_backups.get(index)))
        for scene_index in missing:
            if strategy not in ("mixed", "ai_only") or ai_used >= max_ai_assets:
                break
            prompt = self.image_prompts[scene_index] if scene_index < len(self.image_prompts) else self.image_prompts[-1]
            try:
                if self.generate_image(prompt, scene_index=scene_index):
                    ai_used += 1
            except ImageRateLimitError:
                raise

        def fill_from_stock_backups() -> None:
            # Budget spent: a below-threshold stock clip still beats repeating the previous scene.
            existing = self._get_existing_visual_asset_scene_indices(target_count)
            for scene_index, candidate in sorted(self._stock_backups.items()):
                if scene_index in existing or scene_index >= target_count:
                    continue
                try:
                    if self._download_stock_candidate(candidate, scene_index):
                        existing.add(scene_index)
                except Exception as exc:
                    warning(f"Backup stock clip for scene {scene_index + 1} failed: {exc}")

        fill_from_stock_backups()

        for retry_round in range(1, VISUAL_ASSET_RETRY_ROUNDS + 1):
            existing_scene_indices = self._get_existing_visual_asset_scene_indices(target_count)
            missing_scene_indices = [
                scene_index for scene_index in range(target_count)
                if scene_index not in existing_scene_indices
            ]
            if not missing_scene_indices:
                break

            warning(
                f"Generated {len(existing_scene_indices)}/{target_count} visual assets. "
                f"Retrying {len(missing_scene_indices)} missing scene(s) "
                f"({retry_round}/{VISUAL_ASSET_RETRY_ROUNDS})."
            )

            if strategy in ("mixed", "pixabay_only"):
                retry_scene_units = [scene_units[scene_index] for scene_index in missing_scene_indices]
                retry_prompt_texts = [
                    self.image_prompts[scene_index] if scene_index < len(self.image_prompts) else self.image_prompts[-1]
                    for scene_index in missing_scene_indices
                ]
                retry_durations = [
                    estimated_scene_durations[scene_index] if scene_index < len(estimated_scene_durations) else None
                    for scene_index in missing_scene_indices
                ]
                pixabay_assets_by_scene.update(
                    self._plan_pixabay_scene_assets(
                        retry_scene_units,
                        retry_prompt_texts,
                        retry_durations,
                        scene_indices=missing_scene_indices,
                    )
                )

            previous_count = len(self._get_existing_visual_asset_scene_indices(target_count))

            for scene_index in missing_scene_indices:
                if scene_index in self._get_existing_visual_asset_scene_indices(target_count):
                    continue

                if strategy in ("mixed", "ai_only") and ai_used < max_ai_assets:
                    prompt = self.image_prompts[scene_index] if scene_index < len(self.image_prompts) else self.image_prompts[-1]
                    try:
                        asset_created = bool(self.generate_image(prompt, scene_index=scene_index))
                        if asset_created:
                            ai_used += 1
                    except ImageRateLimitError:
                        raise

            fill_from_stock_backups()
            current_count = len(self._get_existing_visual_asset_scene_indices(target_count))
            if current_count <= previous_count:
                break

        return len(self.visual_assets)

    def _get_video_trim_window(self, clip_duration: float, target_duration: float, scene_index: int) -> tuple[float, float]:
        """
        Chooses a deterministic trim window inside a longer source video.

        Args:
            clip_duration (float): original clip duration
            target_duration (float): desired output duration
            scene_index (int): aligned scene order

        Returns:
            window (tuple[float, float]): start and end times
        """
        clip_duration = max(0.0, float(clip_duration or 0.0))
        target_duration = max(0.1, float(target_duration or 0.1))
        headroom = max(0.0, clip_duration - target_duration)
        if headroom <= 0:
            return 0.0, min(target_duration, clip_duration)

        anchor = ((scene_index % 5) + 1) / 6.0
        start_time = headroom * anchor
        return start_time, start_time + target_duration

    def _normalize_trimmed_pixabay_video(
        self,
        asset_path: str,
        start_time: float,
        duration: float,
        scene_index: int,
    ) -> str | None:
        """
        Re-encodes a trimmed Pixabay video window into a clean H.264/yuv420p
        clip before MoviePy loads it. This avoids intermittent black renders
        seen when seeking into some stock MP4s directly.

        Args:
            asset_path (str): original stock video path
            start_time (float): trim start in seconds
            duration (float): requested duration in seconds
            scene_index (int): aligned scene order

        Returns:
            normalized_path (str | None): safe trimmed clip path when successful
        """
        safe_stem = self._slugify_label(os.path.splitext(os.path.basename(asset_path))[0])
        start_ms = int(round(max(0.0, start_time) * 1000))
        duration_ms = int(round(max(0.1, duration) * 1000))
        normalized_path = self._get_workspace_path(
            f"{safe_stem}_pixabay_scene_{scene_index:02d}_{start_ms}_{duration_ms}.mp4"
        )

        if os.path.exists(normalized_path) and os.path.getsize(normalized_path) > 0:
            return normalized_path

        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            # Loop the source so a clip shorter than the scene fills the whole
            # duration instead of freezing on its last frame.
            "-stream_loop",
            "-1",
            "-i",
            asset_path,
            "-ss",
            f"{max(0.0, start_time):.3f}",
            "-t",
            f"{max(0.1, duration):.3f}",
            "-an",
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                normalized_path,
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if os.path.exists(normalized_path) and os.path.getsize(normalized_path) > 0:
                if get_verbose():
                    info(f' => Normalized Pixabay segment to "{normalized_path}"', show_emoji=False)
                return normalized_path
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            if get_verbose():
                warning(f'Failed to normalize Pixabay clip "{asset_path}": {exc}', show_emoji=False)

        return None

    def _build_visual_clip(self, asset: dict, duration: float):
        """
        Builds a MoviePy clip from an image or stock video asset.

        Args:
            asset (dict): asset descriptor
            duration (float): requested clip duration

        Returns:
            clip: MoviePy clip
        """
        asset_type = asset.get("type")
        asset_path = asset.get("path")
        asset_source = str(asset.get("source") or "").strip().lower()
        scene_index = int(asset.get("scene_index", 0) or 0)

        if asset_type == "video":
            working_asset_path = asset_path
            clip = VideoFileClip(working_asset_path).without_audio()
            if clip.duration >= duration:
                start_time, end_time = self._get_video_trim_window(clip.duration, duration, scene_index)
                trim_duration = max(0.1, end_time - start_time)

                # Pixabay clips occasionally decode fine as full files but render
                # black after an in-memory seek. Normalize the trimmed window first.
                if asset_source in ("pixabay", "pexels") and start_time > 0:
                    clip.close()
                    normalized_asset_path = self._normalize_trimmed_pixabay_video(
                        asset_path,
                        start_time=start_time,
                        duration=trim_duration,
                        scene_index=scene_index,
                    )
                    if normalized_asset_path:
                        working_asset_path = normalized_asset_path
                        clip = VideoFileClip(working_asset_path).without_audio()
                    else:
                        clip = VideoFileClip(asset_path).without_audio().subclip(start_time, end_time)
                else:
                    clip = clip.subclip(start_time, end_time)
            else:
                clip = clip.fx(vfx.loop, duration=duration)
            clip = clip.set_fps(get_video_fps())

            # Crop and resize video to 1080x1920
            if round((clip.w / clip.h), 4) < 0.5625:
                if get_verbose():
                    info(f' => Resizing Asset: {working_asset_path} to 1080x1920')
                clip = crop(
                    clip,
                    width=clip.w,
                    height=round(clip.w / 0.5625),
                    x_center=clip.w / 2,
                    y_center=clip.h / 2,
                )
            else:
                if get_verbose():
                    info(f' => Resizing Asset: {working_asset_path} to 1920x1080')
                clip = crop(
                    clip,
                    width=round(0.5625 * clip.h),
                    height=clip.h,
                    x_center=clip.w / 2,
                    y_center=clip.h / 2,
                )
            clip = clip.resize((1080, 1920))
            clip = self._stabilize_video_scene_clip(clip, duration)
        else:
            # For images, use FFmpeg-based Ken Burns (10-20x faster than MoviePy)
            try:
                kenburns_video_path = self._apply_image_motion_ffmpeg(asset_path, duration, scene_index)
                clip = VideoFileClip(kenburns_video_path).without_audio()
                clip = clip.set_fps(get_video_fps())
            except Exception as e:
                # Fallback to slow MoviePy method if FFmpeg fails
                warning(f"FFmpeg Ken Burns failed, using fallback: {e}")
                clip = ImageClip(asset_path)
                clip = clip.set_duration(duration).set_fps(get_video_fps())

                if round((clip.w / clip.h), 4) < 0.5625:
                    clip = crop(
                        clip,
                        width=clip.w,
                        height=round(clip.w / 0.5625),
                        x_center=clip.w / 2,
                        y_center=clip.h / 2,
                    )
                else:
                    clip = crop(
                        clip,
                        width=round(0.5625 * clip.h),
                        height=clip.h,
                        x_center=clip.w / 2,
                        y_center=clip.h / 2,
                    )
                clip = clip.resize((1080, 1920))
                clip = self._apply_image_motion(clip, duration, scene_index)

        return clip

    def _stabilize_video_scene_clip(self, clip, duration: float):
        """
        Wraps a prepared video scene in a fixed-size composite clip so MoviePy
        does not have to concatenate mixed clip types directly.

        Args:
            clip: cropped/resized MoviePy video clip
            duration (float): requested scene duration

        Returns:
            clip: stabilized composite video clip
        """
        duration = max(0.1, float(duration or 0.1))

        return (
            CompositeVideoClip(
                [clip.set_position(("center", "center"))],
                size=(1080, 1920),
            )
            .set_duration(duration)
            .set_fps(get_video_fps())
        )

    def _apply_image_motion(self, clip, duration: float, scene_index: int):
        """
        Applies a subtle Ken Burns-style move to still images so they feel
        more like living scenes than static slides.

        NOTE: This method is deprecated in favor of _apply_image_motion_ffmpeg
        which is 10-20x faster. Kept for fallback compatibility.

        Args:
            clip: base MoviePy image clip already normalized to 1080x1920
            duration (float): scene duration
            scene_index (int): scene order

        Returns:
            clip: animated image clip
        """
        duration = max(0.35, float(duration or 0.35))
        canvas_w, canvas_h = 1080, 1920

        motion_patterns = [
            {"zoom_start": 1.08, "zoom_end": 1.16, "x_start": 0.00, "x_end": -70.0, "y_start": -10.0, "y_end": 20.0},
            {"zoom_start": 1.15, "zoom_end": 1.05, "x_start": -60.0, "x_end": 20.0, "y_start": -25.0, "y_end": 10.0},
            {"zoom_start": 1.10, "zoom_end": 1.18, "x_start": 25.0, "x_end": -25.0, "y_start": -40.0, "y_end": 40.0},
            {"zoom_start": 1.14, "zoom_end": 1.08, "x_start": -20.0, "x_end": 35.0, "y_start": 15.0, "y_end": -35.0},
            {"zoom_start": 1.07, "zoom_end": 1.14, "x_start": 0.0, "x_end": 0.0, "y_start": -60.0, "y_end": 35.0},
            {"zoom_start": 1.12, "zoom_end": 1.06, "x_start": -35.0, "x_end": 45.0, "y_start": 0.0, "y_end": 0.0},
        ]
        pattern = motion_patterns[scene_index % len(motion_patterns)]

        def lerp(start: float, end: float, progress: float) -> float:
            return start + ((end - start) * progress)

        def current_zoom(t: float) -> float:
            return lerp(
                pattern["zoom_start"],
                pattern["zoom_end"],
                min(1.0, max(0.0, t / duration)),
            )

        max_zoom = max(pattern["zoom_start"], pattern["zoom_end"])
        max_x_drift = max(0.0, ((canvas_w * max_zoom) - canvas_w) / 2)
        max_y_drift = max(0.0, ((canvas_h * max_zoom) - canvas_h) / 2)

        x_start = max(-max_x_drift, min(max_x_drift, pattern["x_start"]))
        x_end = max(-max_x_drift, min(max_x_drift, pattern["x_end"]))
        y_start = max(-max_y_drift, min(max_y_drift, pattern["y_start"]))
        y_end = max(-max_y_drift, min(max_y_drift, pattern["y_end"]))

        animated = clip.resize(lambda t: current_zoom(t))
        animated = animated.set_position(
            lambda t: (
                (canvas_w - (canvas_w * current_zoom(t))) / 2 + lerp(x_start, x_end, min(1.0, max(0.0, t / duration))),
                (canvas_h - (canvas_h * current_zoom(t))) / 2 + lerp(y_start, y_end, min(1.0, max(0.0, t / duration))),
            )
        )

        return CompositeVideoClip([animated], size=(canvas_w, canvas_h)).set_duration(duration).set_fps(get_video_fps())

    def _apply_image_motion_ffmpeg(self, image_path: str, duration: float, scene_index: int) -> str:
        """
        Applies Ken Burns effect using FFmpeg's native zoompan filter.
        This is 10-20x faster than the MoviePy per-frame approach.

        Args:
            image_path (str): path to the source image
            duration (float): scene duration in seconds
            scene_index (int): scene order for pattern selection

        Returns:
            video_path (str): path to the rendered video clip with Ken Burns effect
        """
        duration = max(0.35, float(duration or 0.35))
        fps = get_video_fps()
        total_frames = max(1, int(duration * fps))
        canvas_w, canvas_h = 1080, 1920

        motion_patterns = [
            {"zoom_start": 1.08, "zoom_end": 1.16, "x_start": 0.00, "x_end": -70.0, "y_start": -10.0, "y_end": 20.0},
            {"zoom_start": 1.15, "zoom_end": 1.05, "x_start": -60.0, "x_end": 20.0, "y_start": -25.0, "y_end": 10.0},
            {"zoom_start": 1.10, "zoom_end": 1.18, "x_start": 25.0, "x_end": -25.0, "y_start": -40.0, "y_end": 40.0},
            {"zoom_start": 1.14, "zoom_end": 1.08, "x_start": -20.0, "x_end": 35.0, "y_start": 15.0, "y_end": -35.0},
            {"zoom_start": 1.07, "zoom_end": 1.14, "x_start": 0.0, "x_end": 0.0, "y_start": -60.0, "y_end": 35.0},
            {"zoom_start": 1.12, "zoom_end": 1.06, "x_start": -35.0, "x_end": 45.0, "y_start": 0.0, "y_end": 0.0},
        ]
        pattern = motion_patterns[scene_index % len(motion_patterns)]

        z_start = pattern["zoom_start"]
        z_end = pattern["zoom_end"]
        x_start = pattern["x_start"]
        x_end = pattern["x_end"]
        y_start = pattern["y_start"]
        y_end = pattern["y_end"]

        # Clamp pan values to safe bounds based on max zoom
        max_zoom = max(z_start, z_end)
        max_x_drift = max(0.0, ((canvas_w * max_zoom) - canvas_w) / 2)
        max_y_drift = max(0.0, ((canvas_h * max_zoom) - canvas_h) / 2)
        x_start = max(-max_x_drift, min(max_x_drift, x_start))
        x_end = max(-max_x_drift, min(max_x_drift, x_end))
        y_start = max(-max_y_drift, min(max_y_drift, y_start))
        y_end = max(-max_y_drift, min(max_y_drift, y_end))

        # Build FFmpeg zoompan filter expressions
        # Zoom: linear interpolation from z_start to z_end over total_frames
        z_expr = f"{z_start}+(({z_end}-{z_start})*on/{total_frames})"

        # Pan X: center position with offset, lerping from x_start to x_end
        # FFmpeg zoompan x = (zoomed_width - output_width) / 2 - pan_offset
        x_pan_expr = f"({x_start}+(({x_end}-{x_start})*on/{total_frames}))"
        x_expr = f"(iw*({z_expr})-{canvas_w})/2-({x_pan_expr})"

        # Pan Y: same logic for vertical
        y_pan_expr = f"({y_start}+(({y_end}-{y_start})*on/{total_frames}))"
        y_expr = f"(ih*({z_expr})-{canvas_h})/2-({y_pan_expr})"

        output_path = self._get_workspace_path(f"scene_{scene_index}_kenburns.mp4")

        # Skip if already rendered (e.g. reused asset from scene alignment)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        # Build the filter chain:
        # 1. Scale image to 1080x1920 (crop to fit aspect ratio first)
        # 2. Apply zoompan for Ken Burns effect
        scale_filter = f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,crop={canvas_w}:{canvas_h}"
        zoompan_filter = f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={canvas_w}x{canvas_h}:fps={fps}"

        # Always use libx264 for intermediate Ken Burns clips to avoid
        # exhausting NVENC GPU encoder sessions. NVENC is reserved for the
        # final video export only.
        encoder = "libx264"
        encoder_params = ["-preset", "veryfast"]

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", f"{scale_filter},{zoompan_filter}",
            "-t", str(duration),
            "-c:v", encoder,
            *encoder_params,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        if get_verbose():
            info(f" => Rendering Ken Burns for scene {scene_index} via FFmpeg ({encoder})...")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            warning(f"FFmpeg Ken Burns failed for scene {scene_index}: {e.stderr}")
            raise

        return output_path

    def _get_scene_durations(self, total_duration: float) -> list[float]:
        """
        Computes scene durations by aligning scene units to actual speech timing.
        Uses Whisper word timestamps when available, falls back to text-weight estimation.

        Args:
            total_duration (float): total voiceover duration

        Returns:
            durations (list[float]): per-scene durations
        """
        scene_units = self._get_scene_units()
        if not scene_units:
            return [max(total_duration, 0.1)]

        # TTS word timings (Edge) give exact scene cuts for free.
        spans = self._align_units_to_word_timings(scene_units, self._load_tts_word_timings())
        if spans:
            starts = [0.0] + [start for start, _ in spans[1:]]
            ends = starts[1:] + [total_duration]
            return [max(0.1, end - start) for start, end in zip(starts, ends)]

        # Try Whisper-based alignment for accurate sync
        if hasattr(self, 'tts_path') and self.tts_path and get_stt_provider() == "local_whisper":
            try:
                durations = self._get_scene_durations_from_whisper(total_duration, scene_units)
                if durations:
                    return durations
            except Exception as e:
                warning(f"Whisper scene alignment failed, using text-weight fallback: {e}")

        return self._get_scene_durations_by_weight(total_duration, scene_units)

    def _load_tts_word_timings(self, audio_path: str | None = None) -> list[dict]:
        """
        Loads the [{word, start, end}] sidecar the TTS wrote for the voiceover
        (Edge only), keeping just real words (drops emoji/punctuation tokens).

        Args:
            audio_path (str | None): voiceover path, defaults to self.tts_path

        Returns:
            words (list[dict]): spoken words in order, or [] when unavailable
        """
        path = audio_path or getattr(self, "tts_path", None)
        if not path:
            return []
        try:
            with open(word_timings_path(path), "r", encoding="utf-8") as file:
                words = json.load(file)
        except (OSError, ValueError):
            return []
        return [word for word in words if re.search(r"\w", str(word.get("word", "")))]

    def _align_units_to_word_timings(self, units: list[str], words: list[dict]) -> list[tuple] | None:
        """
        Maps ordered text units (captions or scenes) to (start, end) seconds by
        walking the spoken words in order.

        Args:
            units (list[str]): text units covering the narration in order
            words (list[dict]): spoken words with start/end seconds

        Returns:
            spans (list[tuple] | None): one (start, end) per unit, or None when
            the word counts disagree too much to trust the alignment
        """
        counts = [
            len([token for token in re.findall(r"\S+", str(unit or "")) if re.search(r"\w", token)])
            for unit in units
        ]
        if not words or abs(sum(counts) - len(words)) > max(2, 0.03 * len(words)):
            return None

        spans = []
        index = 0
        for count in counts:
            first = min(index, len(words) - 1)
            last = min(index + max(count, 1) - 1, len(words) - 1)
            spans.append((float(words[first]["start"]), float(words[last]["end"])))
            index += count
        return spans

    def _get_scene_durations_from_whisper(self, total_duration: float, scene_units: list[str]) -> list[float]:
        """
        Uses Whisper word-level timestamps to find where each scene unit
        starts and ends in the actual audio, producing accurate durations.

        Args:
            total_duration (float): total voiceover duration
            scene_units (list[str]): ordered scene text units

        Returns:
            durations (list[float]): per-scene durations aligned to speech, or empty list on failure
        """
        segments, _, _, _ = self._transcribe_with_whisper(
            self.tts_path,
            word_timestamps=True,
        )

        # Collect all words with their timestamps
        words_with_times = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words_with_times.append({
                        "word": str(word.word).strip().lower(),
                        "start": word.start,
                        "end": word.end,
                    })

        if not words_with_times:
            return []

        # Build a normalized word list for each scene unit
        def normalize_words(text: str) -> list[str]:
            return [w.lower().strip(".,!?;:،؛؟…\"'") for w in re.findall(r"\S+", str(text or ""), flags=re.UNICODE) if w.strip()]

        scene_word_lists = [normalize_words(unit) for unit in scene_units]

        # Match scene units to audio timestamps by walking through the word list
        word_idx = 0
        scene_boundaries = []
        for scene_words in scene_word_lists:
            if not scene_words or word_idx >= len(words_with_times):
                scene_boundaries.append(None)
                continue

            start_idx = word_idx
            # Try to find the best matching span of words for this scene
            best_end_idx = min(word_idx + len(scene_words), len(words_with_times)) - 1

            scene_start = words_with_times[start_idx]["start"]
            scene_end = words_with_times[best_end_idx]["end"]
            scene_boundaries.append((scene_start, scene_end))
            word_idx = best_end_idx + 1

        # Convert boundaries to durations
        durations = []
        for i, boundary in enumerate(scene_boundaries):
            if boundary is None:
                durations.append(0.8)
            else:
                durations.append(max(0.8, boundary[1] - boundary[0]))

        # Normalize to match total_duration exactly
        total_raw = sum(durations) or total_duration or 1.0
        durations = [d * (total_duration / total_raw) for d in durations]
        durations[-1] += max(0.0, total_duration - sum(durations))
        return durations

    def _get_scene_durations_by_weight(self, total_duration: float, scene_units: list[str]) -> list[float]:
        """
        Fallback: estimates scene durations from text characteristics.

        Args:
            total_duration (float): total voiceover duration
            scene_units (list[str]): ordered scene text units

        Returns:
            durations (list[float]): per-scene durations
        """
        BASE_WEIGHT = 1.5
        SENTENCE_END_WEIGHT = 1.2
        MID_PAUSE_WEIGHT = 0.5

        def scene_weight(unit: str) -> float:
            text = str(unit or "")
            words = re.findall(r"\S+", text, flags=re.UNICODE)
            sentence_ends = len(re.findall(r"[.!?؟…]", text))
            mid_pauses = len(re.findall(r"[،,؛;:]", text))
            return BASE_WEIGHT + len(words) + (sentence_ends * SENTENCE_END_WEIGHT) + (mid_pauses * MID_PAUSE_WEIGHT)

        total_weight = sum(scene_weight(unit) for unit in scene_units)
        raw_durations = [
            max(0.8, total_duration * (scene_weight(unit) / total_weight))
            for unit in scene_units
        ]
        total_raw = sum(raw_durations) or total_duration or 1.0
        durations = [duration * (total_duration / total_raw) for duration in raw_durations]
        durations[-1] += max(0.0, total_duration - sum(durations))
        return durations

    def _get_scene_aligned_assets(self) -> list[dict]:
        """
        Orders visual assets against the current scene list, reusing the nearest
        available asset when some scenes failed to generate.

        Returns:
            ordered_assets (list[dict]): one asset per scene
        """
        scene_units = self._get_scene_units()
        if not scene_units:
            return self.visual_assets or [
                {"type": "image", "path": image_path, "source": "legacy", "scene_index": index}
                for index, image_path in enumerate(self.images)
            ]

        assets = self.visual_assets or [
            {"type": "image", "path": image_path, "source": "legacy", "scene_index": index}
            for index, image_path in enumerate(self.images)
        ]
        if not assets:
            return []

        assets_by_scene = {}
        fallback_assets = []
        for asset in assets:
            scene_index = asset.get("scene_index")
            if isinstance(scene_index, int):
                assets_by_scene.setdefault(scene_index, asset)
            fallback_assets.append(asset)

        ordered_assets = []
        last_asset = None
        for scene_index in range(len(scene_units)):
            asset = assets_by_scene.get(scene_index)
            if asset is None:
                asset = last_asset or fallback_assets[min(scene_index, len(fallback_assets) - 1)]
            ordered_assets.append(asset)
            last_asset = asset

        return ordered_assets

    def _sync_images_from_visual_assets(self) -> None:
        """
        Keeps the legacy image list aligned with the current visual asset list.

        Returns:
            None
        """
        self.images = [
            asset.get("path")
            for asset in list(getattr(self, "visual_assets", []) or [])
            if isinstance(asset, dict)
            and str(asset.get("type", "")).strip().lower() == "image"
            and str(asset.get("path", "")).strip()
        ]

    def _replace_scene_asset_record(self, scene_index: int, asset_record: dict) -> dict:
        """
        Replaces the tracked asset for one scene and keeps asset caches aligned.

        Args:
            scene_index (int): scene order to replace
            asset_record (dict): new tracked asset payload

        Returns:
            asset_record (dict): normalized stored asset payload
        """
        normalized_record = dict(asset_record or {})
        normalized_record["scene_index"] = int(scene_index)

        retained_assets = [
            asset
            for asset in list(getattr(self, "visual_assets", []) or [])
            if not (
                isinstance(asset, dict)
                and int(asset.get("scene_index", -9999) or -9999) == int(scene_index)
            )
        ]

        insert_index = min(max(int(scene_index), 0), len(retained_assets))
        retained_assets.insert(insert_index, normalized_record)
        self.visual_assets = retained_assets
        self._sync_images_from_visual_assets()
        self._write_workspace_state()
        return normalized_record

    def _parse_srt_entries_utf8(self, srt_path: str) -> list[tuple[tuple[float, float], str]]:
        """
        Loads an SRT file as UTF-8 and converts it into SubtitlesClip entries.

        Args:
            srt_path (str): subtitle file path

        Returns:
            entries (list[tuple[tuple[float, float], str]]): parsed subtitle entries
        """
        with open(srt_path, "r", encoding="utf-8") as file:
            raw = file.read().strip()

        if not raw:
            return []

        blocks = re.split(r"\r?\n\r?\n+", raw)
        entries: list[tuple[tuple[float, float], str]] = []

        for block in blocks:
            lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
            if len(lines) < 3:
                continue

            timing_line = lines[1] if "-->" in lines[1] else lines[0]
            text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
            if "-->" not in timing_line:
                continue

            start_raw, end_raw = [part.strip() for part in timing_line.split("-->", 1)]
            entries.append(((self._srt_timestamp_to_seconds(start_raw), self._srt_timestamp_to_seconds(end_raw)), "\n".join(text_lines)))

        return entries

    def _srt_timestamp_to_seconds(self, value: str) -> float:
        """
        Converts an SRT timestamp into seconds.

        Args:
            value (str): SRT timestamp

        Returns:
            seconds (float): timestamp in seconds
        """
        hours, minutes, seconds = value.split(":")
        secs, millis = seconds.split(",")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(secs)
            + int(millis) / 1000
        )

    def generate_image_nanobanana2(self, prompt: str, scene_index: int | None = None) -> str:
        """
        Generates an AI Image using Nano Banana 2 API (Gemini image API).

        Args:
            prompt (str): Prompt for image generation

        Returns:
            path (str): The path to the generated image.
        """
        print(f"Generating Image using Nano Banana 2 API: {prompt}")

        api_key = get_nanobanana2_api_key()
        if not api_key:
            error("nanobanana2_api_key is not configured.")
            return None

        base_url = get_nanobanana2_api_base_url().rstrip("/")
        model = get_nanobanana2_model()
        aspect_ratio = get_nanobanana2_aspect_ratio()

        endpoint = f"{base_url}/models/{model}:generateContent"

        def build_payload(prompt_text: str) -> dict:
            return {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            }

        prompt_variants = [
            prompt,
            self._make_provider_friendly_image_prompt(prompt),
        ]

        for prompt_index, prompt_variant in enumerate(prompt_variants, start=1):
            payload = build_payload(prompt_variant)
            if prompt_index > 1 and get_verbose():
                warning("Retrying Nano Banana with a shorter provider-friendly prompt.")

            for attempt in range(1, IMAGE_RATE_LIMIT_RETRIES + 2):
                try:
                    response = requests.post(
                        endpoint,
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=get_image_request_timeout(),
                    )

                    if _is_retryable_rate_limit(response):
                        if attempt > IMAGE_RATE_LIMIT_RETRIES:
                            raise ImageRateLimitError(
                                "Nano Banana 2 image generation hit the API rate limit. "
                                "Wait a bit, reduce prompt volume, or use a higher quota before trying again."
                            )

                        retry_after = response.headers.get("Retry-After")
                        wait_seconds = (
                            int(retry_after)
                            if retry_after and retry_after.isdigit()
                            else IMAGE_RATE_LIMIT_BACKOFF_SECONDS * attempt
                        )
                        warning(
                            f"Image API rate-limited. Waiting {wait_seconds}s before retry {attempt}/{IMAGE_RATE_LIMIT_RETRIES}."
                        )
                        time.sleep(wait_seconds)
                        continue

                    # Gemini frequently returns transient 503 "overloaded".
                    # Retry quickly with backoff instead of failing the scene.
                    if _is_retryable_server_error(response) and attempt <= IMAGE_RATE_LIMIT_RETRIES:
                        wait_seconds = IMAGE_RATE_LIMIT_BACKOFF_SECONDS * attempt
                        warning(
                            f"Image API unavailable ({response.status_code}). "
                            f"Waiting {wait_seconds}s before retry {attempt}/{IMAGE_RATE_LIMIT_RETRIES}."
                        )
                        time.sleep(wait_seconds)
                        continue

                    response.raise_for_status()
                    body = response.json()

                    candidates = body.get("candidates", [])
                    for candidate in candidates:
                        content = candidate.get("content", {})
                        for part in content.get("parts", []):
                            inline_data = part.get("inlineData") or part.get("inline_data")
                            if not inline_data:
                                continue
                            data = inline_data.get("data")
                            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
                            if data and str(mime_type).startswith("image/"):
                                image_bytes = base64.b64decode(data)
                                path = self._persist_image(image_bytes, "Nano Banana 2 API", scene_index=scene_index)
                                self._record_image_generation_cost(
                                    "nanobanana2",
                                    get_nanobanana2_model(),
                                    scene_index=scene_index,
                                )
                                return path

                    if _nanobanana_response_has_no_image(body) and prompt_index < len(prompt_variants):
                        break

                    if get_verbose():
                        warning(f"Nano Banana 2 did not return an image payload. Response: {body}")
                    return None
                except Exception as e:
                    if isinstance(e, ImageRateLimitError):
                        raise
                    if get_verbose():
                        warning(f"Failed to generate image with Nano Banana 2 API: {str(e)}")
                    return None

        return None

    def _get_openai_image_size(self) -> str:
        """
        Maps the configured aspect ratio to an OpenAI-supported image size.

        Returns:
            size (str): image size string
        """
        aspect_ratio = get_nanobanana2_aspect_ratio()
        size_map = {
            "1:1": "1024x1024",
            "9:16": "1024x1536",
            "16:9": "1536x1024",
        }
        return size_map.get(aspect_ratio, "1024x1536")

    def _decode_data_url_image(self, data_url: str) -> bytes | None:
        """
        Decodes a data URL containing image bytes.

        Args:
            data_url (str): data URL payload

        Returns:
            image_bytes (bytes | None): decoded image bytes if available
        """
        if not data_url or not isinstance(data_url, str):
            return None

        if not data_url.startswith("data:"):
            return None

        try:
            _, encoded = data_url.split(",", 1)
        except ValueError:
            return None

        return base64.b64decode(encoded)

    def _get_openrouter_modalities(self, model_name: str) -> list[str]:
        """
        Returns the correct OpenRouter modalities for the selected image model.

        Flux and Sourceful models are image-only, while Gemini-style image
        models can return both text and images.

        Args:
            model_name (str): OpenRouter model id

        Returns:
            modalities (list[str]): request modalities
        """
        normalized = str(model_name or "").strip().lower()
        image_only_prefixes = (
            "black-forest-labs/flux",
            "sourceful/",
        )

        if normalized.startswith(image_only_prefixes):
            return ["image"]

        return ["image", "text"]

    def generate_image_openai(self, prompt: str, scene_index: int | None = None) -> str:
        """
        Generates an AI image using OpenAI's Images API.

        Args:
            prompt (str): Prompt for image generation

        Returns:
            path (str): The path to the generated image.
        """
        print(f"Generating Image using OpenAI Images API: {prompt}")

        api_key = get_openai_api_key()
        if not api_key:
            error("openai_api_key is not configured.")
            return None

        endpoint = f"{get_openai_base_url().rstrip('/')}/images/generations"
        payload = {
            "model": get_openai_image_model(),
            "prompt": prompt,
            "size": self._get_openai_image_size(),
            "quality": get_openai_image_quality(),
            "n": 1,
            "output_format": "png",
        }

        for attempt in range(1, IMAGE_RATE_LIMIT_RETRIES + 2):
            try:
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=get_image_request_timeout(),
                )

                if _is_retryable_rate_limit(response):
                    if attempt > IMAGE_RATE_LIMIT_RETRIES:
                        raise ImageRateLimitError(
                            "OpenAI image generation hit the API rate limit. "
                            "Wait a bit, reduce prompt volume, or use a higher quota before trying again."
                        )

                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = (
                        int(retry_after)
                        if retry_after and retry_after.isdigit()
                        else IMAGE_RATE_LIMIT_BACKOFF_SECONDS * attempt
                    )
                    warning(
                        f"OpenAI image API rate-limited. Waiting {wait_seconds}s before retry {attempt}/{IMAGE_RATE_LIMIT_RETRIES}."
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                body = response.json()
                images = body.get("data", [])
                if not images:
                    if get_verbose():
                        warning(f"OpenAI Images API did not return image data. Response: {body}")
                    return None

                b64_image = images[0].get("b64_json")
                if not b64_image:
                    if get_verbose():
                        warning(f"OpenAI Images API returned data without b64_json. Response: {body}")
                    return None

                image_bytes = base64.b64decode(b64_image)
                path = self._persist_image(image_bytes, "OpenAI Images API", scene_index=scene_index)
                self._record_image_generation_cost(
                    "openai",
                    get_openai_image_model(),
                    scene_index=scene_index,
                )
                return path
            except Exception as e:
                if isinstance(e, ImageRateLimitError):
                    raise
                if get_verbose():
                    warning(f"Failed to generate image with OpenAI Images API: {str(e)}")
                return None

    def generate_image_openrouter(self, prompt: str, scene_index: int | None = None) -> str:
        """
        Generates an AI image using OpenRouter's image-capable chat API.

        Args:
            prompt (str): Prompt for image generation

        Returns:
            path (str): The path to the generated image.
        """
        print(f"Generating Image using OpenRouter: {prompt}")

        api_key = get_openrouter_api_key()
        if not api_key:
            error("openrouter_api_key is not configured.")
            return None

        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        model_name = get_openrouter_image_model()
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": self._get_openrouter_modalities(model_name),
            "image_config": {"aspect_ratio": get_nanobanana2_aspect_ratio()},
        }

        for attempt in range(1, IMAGE_RATE_LIMIT_RETRIES + 2):
            try:
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://openrouter.ai",
                        "X-Title": "MoneyPrinterPro",
                    },
                    json=payload,
                    timeout=get_image_request_timeout(),
                )

                if _is_retryable_rate_limit(response):
                    if attempt > IMAGE_RATE_LIMIT_RETRIES:
                        raise ImageRateLimitError(
                            "OpenRouter image generation hit the API rate limit. "
                            "Wait a bit, reduce prompt volume, or use a higher quota before trying again."
                        )

                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = (
                        int(retry_after)
                        if retry_after and retry_after.isdigit()
                        else IMAGE_RATE_LIMIT_BACKOFF_SECONDS * attempt
                    )
                    warning(
                        f"OpenRouter image API rate-limited. Waiting {wait_seconds}s before retry {attempt}/{IMAGE_RATE_LIMIT_RETRIES}."
                    )
                    time.sleep(wait_seconds)
                    continue

                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    error_details = response.text.strip()
                    if error_details:
                        raise RuntimeError(
                            f"OpenRouter image request failed with status {response.status_code}: {error_details}"
                        ) from exc
                    raise

                body = response.json()
                choices = body.get("choices", [])
                if not choices:
                    if get_verbose():
                        warning(f"OpenRouter did not return image choices. Response: {body}")
                    return None

                message = choices[0].get("message", {})
                images = message.get("images", [])
                if not images:
                    if get_verbose():
                        warning(f"OpenRouter did not return images in the response. Response: {body}")
                    return None

                first_image = images[0]
                image_url = first_image.get("url")

                if not image_url:
                    image_url_field = first_image.get("image_url")
                    if isinstance(image_url_field, dict):
                        image_url = image_url_field.get("url")
                    elif isinstance(image_url_field, str):
                        image_url = image_url_field

                if not image_url:
                    image_url_field = first_image.get("imageUrl")
                    if isinstance(image_url_field, dict):
                        image_url = image_url_field.get("url")
                    elif isinstance(image_url_field, str):
                        image_url = image_url_field

                image_bytes = self._decode_data_url_image(image_url)

                if image_bytes is None and image_url:
                    downloaded = requests.get(image_url, timeout=get_image_request_timeout())
                    downloaded.raise_for_status()
                    image_bytes = downloaded.content

                if image_bytes is None:
                    if get_verbose():
                        warning(f"OpenRouter returned an unsupported image payload. Response: {body}")
                    return None

                path = self._persist_image(image_bytes, "OpenRouter", scene_index=scene_index)
                self._record_image_generation_cost(
                    "openrouter",
                    model_name,
                    scene_index=scene_index,
                )
                return path
            except Exception as e:
                if isinstance(e, ImageRateLimitError):
                    raise
                if get_verbose():
                    warning(f"Failed to generate image with OpenRouter: {str(e)}")
                return None

    def generate_image_qwen_local(self, prompt: str, scene_index: int | None = None) -> str:
        """
        Generates an AI image using a self-hosted Qwen-Image server
        (see scripts/qwen_server.py). Costs nothing per image; the server may
        be on this machine or on a rented GPU, set by qwen_api_base_url.

        Args:
            prompt (str): Prompt for image generation
            scene_index (int | None): aligned scene index

        Returns:
            path (str): The path to the generated image.
        """
        print(f"Generating Image using local Qwen-Image server: {prompt}")

        base_url = get_qwen_api_base_url()
        if not base_url:
            error("qwen_api_base_url is not configured.")
            return None

        width, height = get_qwen_dimensions(get_nanobanana2_aspect_ratio())
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": get_qwen_steps(),
        }

        headers = {"Content-Type": "application/json"}
        api_key = get_qwen_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(1, IMAGE_RATE_LIMIT_RETRIES + 2):
            try:
                response = requests.post(
                    f"{base_url}/generate",
                    headers=headers,
                    json=payload,
                    timeout=get_qwen_request_timeout(),
                )

                # A single-GPU server serialises requests, so a busy signal is
                # normal rather than a quota problem worth aborting the run for.
                if _is_retryable_server_error(response) and attempt <= IMAGE_RATE_LIMIT_RETRIES:
                    wait_seconds = IMAGE_RATE_LIMIT_BACKOFF_SECONDS * attempt
                    warning(
                        f"Qwen server unavailable ({response.status_code}). "
                        f"Waiting {wait_seconds}s before retry {attempt}/{IMAGE_RATE_LIMIT_RETRIES}."
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    if get_verbose():
                        warning(
                            f"Qwen server returned {content_type or 'no content type'} "
                            f"instead of an image. Body: {response.text[:400]}"
                        )
                    return None

                path = self._persist_image(response.content, "Qwen-Image (local)", scene_index=scene_index)
                self._record_image_generation_cost(
                    "qwen_local",
                    "qwen-image",
                    scene_index=scene_index,
                )
                return path
            except Exception as e:
                if get_verbose():
                    warning(f"Failed to generate image with the Qwen server: {str(e)}")
                return None

        return None

    def generate_image(self, prompt: str, scene_index: int | None = None) -> str:
        """
        Generates an AI image based on the configured provider.

        Args:
            prompt (str): Reference for image generation

        Returns:
            path (str): The path to the generated image.
        """
        provider = get_image_provider()

        if provider == "openai":
            return self.generate_image_openai(prompt, scene_index=scene_index)

        if provider == "openrouter":
            return self.generate_image_openrouter(prompt, scene_index=scene_index)

        if provider == "qwen_local":
            return self.generate_image_qwen_local(prompt, scene_index=scene_index)

        return self.generate_image_nanobanana2(prompt, scene_index=scene_index)

    def generate_script_to_speech(self, tts_instance: TTS) -> str:
        """
        Converts the generated script into Speech using KittenTTS and returns the path to the wav file.

        Args:
            tts_instance (tts): Instance of TTS Class.

        Returns:
            path_to_wav (str): Path to generated audio (WAV Format).
        """
        path = self._get_workspace_path("voiceover.wav")
        tts_text = str(self.script or "").strip()
        if not tts_text:
            raise RuntimeError("Cannot generate TTS without a script.")

        tts_instance.synthesize(
            tts_text,
            path,
            language=self.language,
            dialect=self.dialect,
            cost_callback=self._record_tts_cost,
        )

        self.tts_path = path

        if get_verbose():
            info(f' => Wrote TTS to "{path}"')
        self._emit_progress(
            "tts",
            "Generated voiceover audio.",
            {"tts_path": path},
        )
        self._write_workspace_state()

        return path

    def add_video(self, video: dict) -> None:
        """
        Adds a video to the cache.

        Args:
            video (dict): The video to add

        Returns:
            None
        """
        cache = get_youtube_cache_path()
        with open(cache, "r", encoding="utf-8-sig") as file:
            previous_json = json.load(file)

            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    account["videos"].append(video)
                    break

            # Commit changes
            with open(cache, "w", encoding="utf-8") as file:
                json.dump(previous_json, file, indent=4, ensure_ascii=False)

    def generate_subtitles(self, audio_path: str) -> str:
        """
        Generates subtitles for the audio using the configured stt_provider.
        Falls back to script-based subtitles only when stt_provider is
        "script_based" or when the chosen provider fails and a script exists.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            path (str): The path to the generated SRT File.
        """
        provider = str(get_stt_provider() or "script_based").lower()

        if provider == "local_whisper":
            try:
                return self.generate_subtitles_local_whisper(audio_path)
            except Exception as e:
                warning(f"Local Whisper failed: {e}")
                if str(self.script or "").strip():
                    warning("Falling back to script-based subtitles.")
                    return self.generate_subtitles_from_script(audio_path)
                raise

        if provider == "third_party_assemblyai":
            try:
                return self.generate_subtitles_assemblyai(audio_path)
            except Exception as e:
                warning(f"AssemblyAI failed: {e}")
                if str(self.script or "").strip():
                    warning("Falling back to script-based subtitles.")
                    return self.generate_subtitles_from_script(audio_path)
                raise

        if provider == "script_based":
            if str(self.script or "").strip():
                return self.generate_subtitles_from_script(audio_path)
            warning("No script available for script_based subtitles. Trying local_whisper.")
            return self.generate_subtitles_local_whisper(audio_path)

        warning(f"Unknown stt_provider '{provider}'. Falling back to local_whisper.")
        return self.generate_subtitles_local_whisper(audio_path)

    def generate_subtitles_assemblyai(self, audio_path: str) -> str:
        """
        Generates subtitles using AssemblyAI.

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        aai.settings.api_key = get_assemblyai_api_key()
        config = aai.TranscriptionConfig()
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        subtitles = transcript.export_subtitles_srt()

        srt_path = self._get_workspace_path("subtitles.srt")

        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        self._record_stt_cost("third_party_assemblyai", audio_path, label="AssemblyAI subtitles")
        return srt_path

    def _get_transcription_language_code(self) -> str | None:
        """
        Maps the configured content language to a transcription language code.

        Returns:
            code (str | None): Whisper-compatible language code or None for auto
        """
        normalized = str(self.language or "").strip().lower()
        language_map = {
            "arabic": "ar",
            "egyptian arabic": "ar",
            "gulf arabic": "ar",
            "levantine arabic": "ar",
            "saudi arabic": "ar",
            "moroccan darija": "ar",
            "darija": "ar",
            "english": "en",
        }
        return language_map.get(normalized)

    def _get_whisper_transcribe_kwargs(self, *, word_timestamps: bool = False) -> dict:
        """
        Builds the standard faster-whisper transcription kwargs for the current
        language configuration.

        Args:
            word_timestamps (bool): whether to request word timestamps

        Returns:
            kwargs (dict): transcription kwargs
        """
        transcription_language = self._get_transcription_language_code()
        transcribe_kwargs = {
            "vad_filter": True,
            "task": "transcribe",
            "word_timestamps": bool(word_timestamps),
        }
        if transcription_language:
            transcribe_kwargs["language"] = transcription_language
        return transcribe_kwargs

    def _transcribe_with_whisper(self, audio_path: str, *, word_timestamps: bool = False):
        """
        Runs faster-whisper with a CPU fallback when the configured backend is
        unavailable or broken.  Results are cached on the instance so that
        multiple callers (scene alignment, subtitles, SFX) share a single
        transcription pass.

        Args:
            audio_path (str): audio file path
            word_timestamps (bool): whether to request word timestamps

        Returns:
            result (tuple[list, object, str, str]): segments, info, device, compute type
        """
        # Return cached result when available and sufficient
        cache = getattr(self, "_whisper_cache", None)
        if cache is not None:
            cached_path, cached_wt, cached_result = cache
            if cached_path == os.path.abspath(audio_path) and (cached_wt or not word_timestamps):
                return cached_result

        from faster_whisper import WhisperModel

        requested_device = get_whisper_device()
        requested_compute_type = get_whisper_compute_type()
        # Always request word_timestamps so the cache satisfies all callers
        transcribe_kwargs = self._get_whisper_transcribe_kwargs(
            word_timestamps=True,
        )

        def attempt(device: str, compute_type: str):
            model = WhisperModel(
                get_whisper_model(),
                device=device,
                compute_type=compute_type,
            )
            segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)
            return list(segments_iter), info

        try:
            segments, info = attempt(requested_device, requested_compute_type)
            result = (segments, info, requested_device, requested_compute_type)
        except Exception as primary_exc:
            if requested_device == "cpu" and requested_compute_type == "int8":
                raise

            warning(
                f'Whisper backend "{requested_device}/{requested_compute_type}" failed. '
                f'Falling back to CPU int8: {primary_exc}'
            )
            self._emit_progress(
                "subtitles",
                "Configured Whisper backend failed, retrying on CPU.",
                {
                    "requested_device": requested_device,
                    "requested_compute_type": requested_compute_type,
                },
            )
            segments, info = attempt("cpu", "int8")
            result = (segments, info, "cpu", "int8")

        self._whisper_cache = (os.path.abspath(audio_path), True, result)
        return result

    def _contains_arabic_text(self, value: str) -> bool:
        """
        Returns whether the provided text contains Arabic script.

        Args:
            value (str): text to inspect

        Returns:
            is_arabic (bool): True when Arabic characters are present
        """
        return bool(
            re.search(
                r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
                str(value or ""),
            )
        )

    def _should_use_arabic_subtitle_rendering(self, text: str) -> bool:
        """
        Returns whether subtitles should be rendered with the Arabic font and
        RTL-aware renderer.

        Args:
            text (str): subtitle text

        Returns:
            should_use (bool): True when Arabic rendering should be used
        """
        if self._contains_arabic_text(text):
            return True

        if self._contains_arabic_text(getattr(self, "script", "")):
            return True

        normalized_language = str(getattr(self, "language", "") or "").strip().lower()
        normalized_dialect = str(getattr(self, "dialect", "") or "").strip().lower()
        arabic_markers = (
            "arabic",
            "egyptian",
            "gulf",
            "saudi",
            "levantine",
            "darija",
            "moroccan",
        )
        return any(marker in normalized_language for marker in arabic_markers) or any(
            marker in normalized_dialect for marker in arabic_markers
        )

    def _get_subtitle_font_filename_for_text(self, text: str) -> str:
        """
        Resolves the subtitle font filename based on the subtitle text.

        Args:
            text (str): subtitle text

        Returns:
            font_filename (str): best subtitle font filename
        """
        preferred_font = (
            get_subtitle_font_arabic()
            if self._should_use_arabic_subtitle_rendering(text)
            else get_subtitle_font_english()
        )
        preferred_path = self._resolve_subtitle_font_path(preferred_font)
        if preferred_path:
            return preferred_path

        fallback_font = get_subtitle_font()
        fallback_path = self._resolve_subtitle_font_path(fallback_font)
        if fallback_path:
            return fallback_path

        legacy_path = self._resolve_subtitle_font_path(get_font())
        return legacy_path

    def _resolve_subtitle_font_path(self, font_name: str) -> str:
        """
        Resolves a subtitle font filename/path from repo fonts or common
        system font directories.

        Args:
            font_name (str): configured font name or path

        Returns:
            path (str): resolved absolute font path or original value
        """
        raw_font = str(font_name or "").strip()
        if not raw_font:
            return ""
        if os.path.isabs(raw_font) and os.path.exists(raw_font):
            return os.path.abspath(raw_font)

        candidate_paths = [
            os.path.join(get_fonts_dir(), raw_font),
        ]
        if os.name == "nt":
            candidate_paths.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", raw_font))

        for path in candidate_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        return raw_font

    def _get_explicit_arabic_subtitle_font_path(self) -> str:
        """
        Returns the configured Arabic subtitle font path without falling back
        to the English font selector.

        Returns:
            path (str): Arabic subtitle font path
        """
        preferred_font = get_subtitle_font_arabic()
        preferred_path = self._resolve_subtitle_font_path(preferred_font)
        if preferred_path and os.path.exists(preferred_path):
            return preferred_path

        fallback_font = get_subtitle_font()
        fallback_path = self._resolve_subtitle_font_path(fallback_font)
        if fallback_path and os.path.exists(fallback_path):
            warning(
                f"Configured Arabic subtitle font '{preferred_font}' was not found. Falling back to '{fallback_font}'."
            )
            return fallback_path

        legacy_path = self._resolve_subtitle_font_path(get_font())
        return legacy_path

    def _shape_subtitle_text_for_rendering(self, text: str) -> str:
        """
        Normalizes subtitle text before drawing.

        Args:
            text (str): subtitle text

        Returns:
            shaped (str): render-ready subtitle text
        """
        raw_text = str(text or "").strip()
        return re.sub(r"\s+", " ", raw_text).strip()

    def _pil_supports_rtl_layout(self) -> bool:
        """
        Returns whether the local Pillow build can render RTL scripts natively.

        Returns:
            supports_rtl (bool): True when libraqm-based RTL support is available
        """
        try:
            return bool(pil_features.check("raqm"))
        except Exception:
            return False

    def _shape_arabic_text_for_pil(self, text: str) -> str:
        """
        Shapes Arabic text into visual order for Pillow builds that do not
        support RTL layout natively.

        Args:
            text (str): logical-order subtitle text

        Returns:
            shaped (str): visual-order Arabic text for drawing
        """
        return self._fallback_rtl_visual_order(text)

    def _prepare_text_for_pil(self, text: str) -> tuple[str, dict]:
        """
        Prepares subtitle text for Pillow drawing/measurement.

        Args:
            text (str): logical-order subtitle text

        Returns:
            prepared (tuple[str, dict]): display text and optional draw kwargs
        """
        normalized = self._shape_subtitle_text_for_rendering(text)
        if not normalized:
            return "", {}

        if self._should_use_arabic_subtitle_rendering(normalized):
            if self._pil_supports_rtl_layout():
                return normalized, {"direction": "rtl", "language": "ar"}
            return self._fallback_rtl_visual_order(normalized), {}

        return normalized, {}

    def _get_pil_subtitle_draw_kwargs(self, text: str) -> dict:
        """
        Returns PIL drawing kwargs for subtitle rendering.

        Args:
            text (str): subtitle text

        Returns:
            kwargs (dict): PIL text drawing options
        """
        return self._prepare_text_for_pil(text)[1]

    def _fallback_rtl_visual_order(self, text: str) -> str:
        """
        Fallback for environments where RTL drawing support is unavailable.
        It reverses each line so Arabic can at least stay legible in basic
        left-to-right renderers when no proper RTL engine is available.

        Args:
            text (str): subtitle text

        Returns:
            reordered (str): visually reordered text
        """
        lines = str(text or "").splitlines() or [str(text or "")]
        normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
        normalized_lines = [line for line in normalized_lines if line]
        return "\n".join(line[::-1] for line in normalized_lines)

    def _load_subtitle_font(self, text: str):
        """
        Loads the configured subtitle font for the given text.

        Args:
            text (str): subtitle text

        Returns:
            font (ImageFont.FreeTypeFont): loaded font
        """
        font_path = self._get_subtitle_font_filename_for_text(text)
        font_size = max(24, int(get_subtitle_font_size()))
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_subtitle_text(
        self,
        text: str,
        font,
        max_width: int,
        stroke_width: int,
    ) -> tuple[str, dict]:
        """
        Wraps subtitle text to fit a target width using PIL measurements.

        Args:
            text (str): subtitle text
            font: PIL font
            max_width (int): maximum rendered width
            stroke_width (int): subtitle stroke width

        Returns:
            wrapped (tuple[str, dict]): wrapped subtitle text and draw kwargs
        """
        raw = str(text or "").strip()
        if not raw:
            return "", {}

        drawer = ImageDraw.Draw(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
        words = raw.split()
        if len(words) <= 1:
            prepared_raw, draw_kwargs = self._prepare_text_for_pil(raw)
            try:
                drawer.multiline_textbbox(
                    (0, 0),
                    prepared_raw,
                    font=font,
                    stroke_width=stroke_width,
                    spacing=10,
                    align="center",
                    **draw_kwargs,
                )
                return prepared_raw, draw_kwargs
            except Exception:
                fallback = self._prepare_text_for_pil(raw)[0] or self._fallback_rtl_visual_order(raw)
                return fallback, {}

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}".strip()
            prepared_candidate, draw_kwargs = self._prepare_text_for_pil(candidate)
            try:
                bbox = drawer.multiline_textbbox(
                    (0, 0),
                    prepared_candidate,
                    font=font,
                    stroke_width=stroke_width,
                    spacing=10,
                    align="center",
                    **draw_kwargs,
                )
            except Exception:
                draw_kwargs = {}
                prepared_candidate = self._fallback_rtl_visual_order(candidate)
                bbox = drawer.multiline_textbbox(
                    (0, 0),
                    prepared_candidate,
                    font=font,
                    stroke_width=stroke_width,
                    spacing=10,
                    align="center",
                )
            if (bbox[2] - bbox[0]) <= max_width:
                current = candidate
                continue
            lines.append(current)
            current = word

        if current:
            lines.append(current)

        wrapped = "\n".join(lines)
        prepared_wrapped, draw_kwargs = self._prepare_text_for_pil(wrapped)
        if not prepared_wrapped and self._contains_arabic_text(raw):
            prepared_wrapped = self._fallback_rtl_visual_order(wrapped)
            draw_kwargs = {}
        return prepared_wrapped or wrapped, draw_kwargs

    def _create_subtitle_bitmap(self, text: str) -> np.ndarray:
        """
        Renders one subtitle unit into an RGBA bitmap using PIL so the chosen
        local font files are actually respected.

        Args:
            text (str): subtitle text

        Returns:
            image (np.ndarray): RGBA subtitle bitmap
        """
        shaped_text = self._shape_subtitle_text_for_rendering(text)
        stroke_width = max(0, int(get_subtitle_stroke_width()))
        text_image = self._render_subtitle_text_image(
            shaped_text,
            max_width=900,
            stroke_width=stroke_width,
        )
        text_width = max(1, int(text_image.width))
        text_height = max(1, int(text_image.height))
        pad_x = 48
        pad_y = 30
        image = Image.new(
            "RGBA",
            (text_width + pad_x * 2, text_height + pad_y * 2),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (0, 0, image.width - 1, image.height - 1),
            radius=28,
            fill=(0, 0, 0, 150),
        )
        image.alpha_composite(
            text_image,
            (
                int((image.width - text_image.width) / 2),
                int((image.height - text_image.height) / 2),
            ),
        )
        return np.array(image)

    def _render_subtitle_text_image(
        self,
        text: str,
        max_width: int,
        stroke_width: int,
    ) -> Image.Image:
        """
        Renders subtitle text to a transparent image. Arabic text uses
        ImageMagick for proper shaping and RTL layout; other text uses PIL.

        Args:
            text (str): subtitle text
            max_width (int): maximum text width before wrapping
            stroke_width (int): outline thickness

        Returns:
            image (Image.Image): transparent RGBA text image
        """
        if self._should_use_arabic_subtitle_rendering(text):
            try:
                return self._render_arabic_subtitle_with_imagemagick(
                    text,
                    max_width=max_width,
                    stroke_width=stroke_width,
                )
            except Exception as exc:
                warning(f"Falling back to PIL subtitle rendering for Arabic text: {exc}")

        font = self._load_subtitle_font(text)
        wrapped_text, draw_kwargs = self._wrap_subtitle_text(
            text,
            font,
            max_width=max_width,
            stroke_width=stroke_width,
        )

        drawer = ImageDraw.Draw(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
        bbox = drawer.multiline_textbbox(
            (0, 0),
            wrapped_text,
            font=font,
            stroke_width=stroke_width,
            spacing=12,
            align="center",
            **draw_kwargs,
        )
        text_width = max(1, int(bbox[2] - bbox[0]))
        text_height = max(1, int(bbox[3] - bbox[1]))
        text_image = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)
        draw.multiline_text(
            (text_width / 2, text_height / 2),
            wrapped_text,
            font=font,
            fill=get_subtitle_color(),
            stroke_fill=get_subtitle_stroke_color(),
            stroke_width=stroke_width,
            anchor="mm",
            align="center",
            spacing=12,
            **draw_kwargs,
        )
        return text_image

    def _render_arabic_subtitle_with_imagemagick(
        self,
        text: str,
        max_width: int,
        stroke_width: int,
    ) -> Image.Image:
        """
        Uses ImageMagick's Arabic-capable text rendering delegates to draw
        Arabic subtitles with connected letters and proper RTL shaping.

        Args:
            text (str): subtitle text
            max_width (int): text wrapping width
            stroke_width (int): outline thickness

        Returns:
            image (Image.Image): transparent RGBA text image
        """
        magick_path = get_imagemagick_path()
        font_path = self._get_subtitle_font_filename_for_text(text)
        # Extract font family name for Pango (e.g., "Tahoma" from "C:\Windows\Fonts\tahoma.ttf")
        # Pango needs font family names, not file paths, for proper Arabic text shaping
        font_basename = os.path.basename(font_path)
        font_name_no_ext = os.path.splitext(font_basename)[0]
        # Capitalize first letter for font family name (tahoma -> Tahoma)
        font_family = font_name_no_ext.capitalize() if font_name_no_ext else "Tahoma"

        point_size = max(24, int(get_subtitle_font_size()))
        rgba_fill = str(get_subtitle_color())
        render_text = self._shape_subtitle_text_for_rendering(text)

        # Use Pango for proper Arabic text shaping (connected letters, RTL)
        # Pango handles bidirectional text and Arabic letter joining correctly
        pango_args = [
            magick_path,
            "-background",
            "none",
            "-fill",
            rgba_fill,
            "-font",
            font_family,
            "-pointsize",
            str(point_size),
            "-gravity",
            "center",
            "-size",
            f"{max_width}x",
            f"pango:{render_text}",
            "png:-",
        ]

        try:
            wrapped = subprocess.run(pango_args, capture_output=True, check=True)
            return Image.open(io.BytesIO(wrapped.stdout)).convert("RGBA")
        except subprocess.CalledProcessError as e:
            # Fallback to caption: if pango fails (e.g., Pango not available)
            warning(f"Pango rendering failed, falling back to caption: {e.stderr}")
            font_path_normalized = font_path.replace("\\", "/")
            caption_args = [
                magick_path,
                "-background",
                "none",
                "-fill",
                rgba_fill,
                "-font",
                font_path_normalized,
                "-pointsize",
                str(point_size),
                "-gravity",
                "center",
                "-size",
                f"{max_width}x",
                f"caption:{render_text}",
                "png:-",
            ]
            wrapped = subprocess.run(caption_args, capture_output=True, check=True)
            return Image.open(io.BytesIO(wrapped.stdout)).convert("RGBA")

    def _hex_to_ass_color(self, hex_color: str, alpha: int = 0) -> str:
        """
        Converts a hex color (#RRGGBB) to ASS format (&HAABBGGRR).

        Args:
            hex_color (str): hex color like #FFF7D6
            alpha (int): alpha value 0-255 (0=opaque, 255=transparent)

        Returns:
            ass_color (str): ASS color format &HAABBGGRR
        """
        hex_color = str(hex_color or "#FFFFFF").strip().lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        if len(hex_color) != 6:
            hex_color = "FFFFFF"

        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        # ASS format is &HAABBGGRR (alpha, blue, green, red)
        return f"&H{alpha:02X}{b}{g}{r}"

    def _generate_ass_subtitles(
        self,
        subtitle_entries: list[tuple[tuple[float, float], str]],
        output_path: str,
    ) -> str:
        """
        Generates an ASS (Advanced SubStation Alpha) subtitle file from entries.
        ASS format supports custom fonts, colors, outlines, and proper RTL text.

        Args:
            subtitle_entries (list): timed subtitle entries
            output_path (str): output ASS file path

        Returns:
            path (str): path to generated ASS file
        """
        # Get styling from config
        font_size = max(24, int(get_subtitle_font_size()))
        primary_color = self._hex_to_ass_color(get_subtitle_color(), alpha=0)
        outline_color = self._hex_to_ass_color(get_subtitle_stroke_color(), alpha=0)
        back_color = self._hex_to_ass_color("#000000", alpha=150)  # Semi-transparent background
        outline_width = max(0, int(get_subtitle_stroke_width()))

        # Get font family name
        sample_text = subtitle_entries[0][1] if subtitle_entries else ""
        font_path = self._get_subtitle_font_filename_for_text(sample_text)
        font_basename = os.path.basename(font_path)
        font_name = os.path.splitext(font_basename)[0].capitalize()
        if not font_name:
            font_name = "Tahoma"

        # Position from bottom (1920 - desired_y_position)
        # word_by_word uses y=1510, chunk uses y=1450
        subtitle_y = 1510 if get_subtitle_mode() == "word_by_word" else 1450
        margin_v = 1920 - subtitle_y - 50  # Margin from bottom

        # ASS header
        ass_content = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},-1,0,0,0,100,100,0,0,1,{outline_width},0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Convert entries to ASS dialogue lines
        for (start_time, end_time), text in subtitle_entries:
            start_ass = self._seconds_to_ass_timestamp(start_time)
            end_ass = self._seconds_to_ass_timestamp(end_time)
            # Escape special ASS characters and normalize text
            clean_text = str(text or "").strip()
            clean_text = clean_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            clean_text = clean_text.replace("\n", "\\N")
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{clean_text}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        return output_path

    def _seconds_to_ass_timestamp(self, seconds: float) -> str:
        """
        Converts seconds to ASS timestamp format (H:MM:SS.cc).

        Args:
            seconds (float): time in seconds

        Returns:
            timestamp (str): ASS format timestamp
        """
        seconds = max(0.0, float(seconds or 0.0))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds * 100) % 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _burn_subtitles_with_ffmpeg(
        self,
        video_path: str,
        ass_path: str,
        output_path: str,
    ) -> str:
        """
        Burns ASS subtitles into video using FFmpeg's subtitle filter.
        This is much faster than MoviePy's per-frame overlay compositing.

        Args:
            video_path (str): input video path
            ass_path (str): ASS subtitle file path
            output_path (str): output video path

        Returns:
            path (str): path to output video with burned subtitles
        """
        # Normalize paths for FFmpeg on Windows (use forward slashes and escape colons)
        ass_path_normalized = ass_path.replace("\\", "/")
        # FFmpeg subtitles filter needs special escaping for Windows paths
        # Colons in drive letters need escaping, and backslashes need double escaping
        ass_path_escaped = ass_path_normalized.replace(":", "\\:")

        # Determine encoder
        encoder = "libx264"
        encoder_params = ["-preset", "veryfast"]
        if self._ffmpeg_supports_encoder("h264_nvenc"):
            encoder = "h264_nvenc"
            encoder_params = []

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{ass_path_escaped}'",
            "-c:v", encoder,
            *encoder_params,
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        if get_verbose():
            info(f" => Burning subtitles into video via FFmpeg ({encoder})...")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            warning(f"FFmpeg subtitle burn failed: {e.stderr}")
            raise

        return output_path

    def _build_subtitle_overlay_clips(
        self,
        subtitle_entries: list[tuple[tuple[float, float], str]],
    ) -> list[ImageClip]:
        """
        Converts parsed subtitle entries into positioned overlay ImageClips.

        Args:
            subtitle_entries (list[tuple[tuple[float, float], str]]): timed subtitles

        Returns:
            overlay_clips (list[ImageClip]): subtitle overlay clips
        """
        overlay_clips = []
        subtitle_y = 1510 if get_subtitle_mode() == "word_by_word" else 1450

        for (start_time, end_time), text in subtitle_entries:
            duration = max(0.05, float(end_time) - float(start_time))
            rgba = self._create_subtitle_bitmap(text)
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3].astype("float32") / 255.0
            clip = (
                ImageClip(rgb)
                .set_mask(ImageClip(alpha, ismask=True))
                .set_start(float(start_time))
                .set_duration(duration)
                .set_pos(("center", subtitle_y))
            )
            overlay_clips.append(clip)

        return overlay_clips

    def _should_equalize_subtitles(self) -> bool:
        """
        Returns whether subtitle equalization should be applied.

        Arabic subtitles tend to degrade when aggressively reflowed into
        extremely short lines, so we skip equalization for Arabic content.

        Returns:
            should_equalize (bool): True when equalization is safe/useful
        """
        normalized = str(self.language or "").strip().lower()
        return "arabic" not in normalized and "darija" not in normalized

    def _should_use_script_subtitles(self) -> bool:
        """
        Returns whether subtitles should be generated directly from the script.

        Returns:
            use_script_subtitles (bool): True when script subtitles are preferred
        """
        return True

    def _split_script_into_subtitle_chunks(self) -> list[str]:
        """
        Splits the generated script into subtitle-sized chunks.

        Returns:
            chunks (list[str]): subtitle text chunks
        """
        raw_parts = re.split(r"(?<=[\.\!\?؟،,:;])\s+|\n+", str(self.script or "").strip())
        sentence_parts = [part.strip(" \t\r\n-") for part in raw_parts if part.strip()]

        if not sentence_parts:
            sentence_parts = [str(self.script or "").strip()]

        # Cap each subtitle to a few words so long (often Arabic) sentences don't
        # fill the screen. 0 = no cap (split by sentence only).
        max_words = get_subtitle_max_chunk_words()
        chunks: list[str] = []
        for part in sentence_parts:
            words = part.split()
            if not words:
                continue
            if max_words <= 0 or len(words) <= max_words:
                chunks.append(part)
                continue
            for index in range(0, len(words), max_words):
                piece = " ".join(words[index:index + max_words]).strip()
                if piece:
                    chunks.append(piece)

        return chunks

    def _split_script_into_subtitle_words(self) -> list[str]:
        """
        Splits the generated script into word-by-word subtitle units.

        Returns:
            words (list[str]): subtitle words
        """
        return re.findall(r"\S+", str(self.script or "").strip(), flags=re.UNICODE)

    def _get_subtitle_script_units(self) -> list[str]:
        """
        Returns subtitle timing units based on the configured subtitle mode.

        Returns:
            units (list[str]): subtitle units
        """
        if get_subtitle_mode() == "word_by_word":
            words = self._split_script_into_subtitle_words()
            if words:
                return words

        return self._split_script_into_subtitle_chunks()

    def generate_subtitles_from_script(self, audio_path: str) -> str:
        """
        Generates subtitles directly from the script text.

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        subtitle_units = self._get_subtitle_script_units()
        if not subtitle_units:
            raise RuntimeError("Cannot generate script-based subtitles without a script.")

        audio_clip = AudioFileClip(audio_path)
        total_duration = max(audio_clip.duration, 0.1)
        total_chars = sum(len(unit) for unit in subtitle_units) or len(subtitle_units)
        # Exact per-word timings from the TTS (Edge) beat character-count guessing,
        # which drifts later with every pause in the narration.
        spans = self._align_units_to_word_timings(subtitle_units, self._load_tts_word_timings(audio_path))

        lines = []
        current_time = 0.0
        min_duration = 0.1 if get_subtitle_mode() == "word_by_word" else 0.8
        for idx, unit in enumerate(subtitle_units, start=1):
            if spans:
                # Each caption holds until the next spoken unit begins (no flicker in pauses).
                current_time = spans[idx - 1][0]
                end_time = spans[idx][0] if idx < len(spans) else total_duration
            else:
                unit_weight = len(unit) / total_chars if total_chars else 1 / len(subtitle_units)
                duration = max(total_duration * unit_weight, min_duration)
                end_time = min(total_duration, current_time + duration)

            if idx == len(subtitle_units):
                end_time = total_duration

            lines.append(str(idx))
            lines.append(
                f"{self._format_srt_timestamp(current_time)} --> {self._format_srt_timestamp(end_time)}"
            )
            lines.append(unit)
            lines.append("")
            current_time = end_time

        subtitles = "\n".join(lines)
        srt_path = self._get_workspace_path("subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        audio_clip.close()
        self._record_stt_cost("script_based", audio_path, label="Script-based subtitles")
        return srt_path

    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Formats a timestamp in seconds to SRT format.

        Args:
            seconds (float): Seconds

        Returns:
            ts (str): HH:MM:SS,mmm
        """
        total_millis = max(0, int(round(seconds * 1000)))
        hours = total_millis // 3600000
        minutes = (total_millis % 3600000) // 60000
        secs = (total_millis % 60000) // 1000
        millis = total_millis % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_subtitles_local_whisper(self, audio_path: str) -> str:
        """
        Generates subtitles using local Whisper (faster-whisper).

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            error(
                "Local STT selected but 'faster-whisper' is not installed. "
                "Install it or switch stt_provider to third_party_assemblyai."
            )
            raise

        self._emit_progress(
            "subtitles_start",
            "Generating subtitles with local Whisper...",
            {
                "provider": "local_whisper",
                "requested_device": get_whisper_device(),
                "requested_compute_type": get_whisper_compute_type(),
            },
        )
        segments, _, used_device, used_compute_type = self._transcribe_with_whisper(
            audio_path,
            word_timestamps=False,
        )

        lines = []
        for idx, segment in enumerate(segments, start=1):
            start = self._format_srt_timestamp(segment.start)
            end = self._format_srt_timestamp(segment.end)
            text = str(segment.text).strip()

            if not text:
                continue

            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        subtitles = "\n".join(lines)
        srt_path = self._get_workspace_path("subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        self._emit_progress(
            "subtitles",
            f"Generated subtitles with Whisper on {used_device}/{used_compute_type}.",
            {"subtitle_entries": len(lines) // 4, "subtitles_path": srt_path},
        )
        self._record_stt_cost("local_whisper", audio_path, get_whisper_model(), "Local Whisper subtitles")
        return srt_path

    def combine(self) -> str:
        """
        Combines everything into the final video. Uses the memory-light
        streaming path by default; falls back to the legacy in-memory path
        when youtube_low_memory_render is disabled.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        if get_youtube_low_memory_render():
            return self._combine_streaming()
        return self._combine_in_memory()

    def _render_scene_files(self, assets: list, scene_durations: list) -> list:
        """
        Renders each scene to its own 1080x1920 MP4 file, one at a time, closing
        each clip before the next so only one scene is ever held in memory.

        Args:
            assets (list): scene-aligned visual assets
            scene_durations (list): per-scene durations

        Returns:
            scene_paths (list[str]): rendered per-scene file paths
        """
        fps = get_video_fps()
        crossfade = get_crossfade_duration()
        total = len(assets)
        scene_paths: list[str] = []

        for index, (asset, scene_duration) in enumerate(zip(assets, scene_durations)):
            self._emit_progress(
                "render_scene",
                f"Rendering scene {index + 1}/{total}...",
                {"scene": index + 1, "total": total},
            )
            clip = self._build_visual_clip(asset, scene_duration)
            if crossfade > 0 and total > 1:
                if index > 0:
                    clip = clip.fadein(crossfade)
                if index < total - 1:
                    clip = clip.fadeout(crossfade)

            scene_path = self._get_workspace_path(f"scene_final_{index:02d}.mp4")
            # Identical libx264/yuv420p/fps params across scenes so they can be
            # concatenated losslessly with the FFmpeg concat demuxer (-c copy).
            clip.write_videofile(
                scene_path,
                fps=fps,
                codec="libx264",
                preset="veryfast",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
                audio=False,
                logger=None,
            )
            try:
                clip.close()
            except Exception:
                pass
            scene_paths.append(scene_path)

        return scene_paths

    def _ffmpeg_concat_videos(self, scene_paths: list, output_path: str) -> str:
        """
        Concatenates same-codec scene files with the FFmpeg concat demuxer.
        Falls back to a re-encode if a stream-copy concat fails.

        Args:
            scene_paths (list[str]): per-scene MP4 files in order
            output_path (str): destination path

        Returns:
            output_path (str): the concatenated video path
        """
        list_path = self._get_workspace_path("scenes_concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in scene_paths:
                safe = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
                handle.write(f"file '{safe}'\n")

        base_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path]
        try:
            subprocess.run(base_cmd + ["-c", "copy", output_path], check=True)
        except subprocess.CalledProcessError:
            # Streams weren't copy-compatible — re-encode the concatenation.
            subprocess.run(
                base_cmd
                + ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", output_path],
                check=True,
            )
        return output_path

    def _build_final_audio_file(self, tts_clip) -> str:
        """
        Mixes voiceover + background song (ducked) + sound effects into a single
        audio file. Audio is light on memory, unlike the video frames.

        Args:
            tts_clip: the voiceover AudioFileClip

        Returns:
            audio_path (str): path to the mixed audio WAV
        """
        audio_layers = [tts_clip.set_fps(44100)]

        try:
            random_song = choose_random_song()
            random_song_clip = AudioFileClip(random_song).set_fps(44100)
            random_song_clip = self._apply_audio_ducking(
                tts_clip, random_song_clip, voice_vol=0.15, silence_vol=0.4
            )
            audio_layers.append(random_song_clip)
        except Exception as exc:
            warning(f"Failed to add background song, continuing with voiceover only: {exc}")

        if get_sound_effects_enabled():
            try:
                sfx_clips = self._build_sound_effects_clips(self.tts_path)
                if sfx_clips:
                    audio_layers.extend(sfx_clips)
                    info(f" => Added {len(sfx_clips)} sound effects to the audio mix.")
            except Exception as exc:
                warning(f"Failed to add sound effects: {exc}")

        final_audio = CompositeAudioClip(audio_layers).set_duration(tts_clip.duration)
        audio_path = self._get_workspace_path("final_audio.wav")
        final_audio.write_audiofile(audio_path, fps=44100, logger=None)
        return audio_path

    def _mux_video_audio(self, video_path: str, audio_path: str, output_path: str, duration: float) -> str:
        """
        Muxes an audio track onto a silent video without re-encoding the video.

        Args:
            video_path (str): silent video
            audio_path (str): audio track
            output_path (str): destination
            duration (float): clamp output to this duration (seconds)

        Returns:
            output_path (str)
        """
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac",
                "-t", f"{max(0.1, float(duration)):.3f}",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
        )
        return output_path

    def _combine_streaming(self) -> str:
        """
        Memory-light final assembly: render each scene to a file, concatenate
        with FFmpeg, mix audio to a file, mux, then burn subtitles. Keeps memory
        flat regardless of scene count.

        Returns:
            path (str): the final MP4 path
        """
        assets = self._get_scene_aligned_assets()
        if len(assets) == 0:
            raise RuntimeError(
                "No visual assets were generated, so the video cannot be combined."
            )

        combined_image_path = self._get_workspace_path("final_video.mp4")
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        scene_durations = self._get_scene_durations(max_duration)

        print(colored("[+] Combining visual assets...", "blue"))
        self._emit_progress("combine", "Combining visual assets into the final timeline.")

        # 1) Render each scene to its own file and stitch them (constant memory).
        scene_paths = self._render_scene_files(assets, scene_durations)
        silent_video_path = self._get_workspace_path("scenes_concat.mp4")
        self._ffmpeg_concat_videos(scene_paths, silent_video_path)

        # 2) Subtitles (ASS for fast FFmpeg burning).
        ass_subtitles_path = None
        try:
            subtitles_path = self.generate_subtitles(self.tts_path)
            if self._should_equalize_subtitles():
                equalize_subtitles(subtitles_path, 18)
            subtitle_entries = self._parse_srt_entries_utf8(subtitles_path)
            if not subtitle_entries:
                raise RuntimeError("Subtitle file was empty after generation.")
            ass_subtitles_path = self._get_workspace_path("subtitles.ass")
            self._generate_ass_subtitles(subtitle_entries, ass_subtitles_path)
            self._emit_progress(
                "subtitles",
                f"Prepared {len(subtitle_entries)} subtitle entries for FFmpeg burning.",
                {"subtitle_entries": len(subtitle_entries), "subtitles_path": ass_subtitles_path},
            )
        except Exception as exc:
            warning(f"Failed to generate subtitles, continuing without subtitles: {exc}")
            ass_subtitles_path = None

        # 3) Mix audio to a file and mux it onto the silent video.
        audio_path = self._build_final_audio_file(tts_clip)
        muxed_path = self._get_workspace_path("video_no_subs.mp4")
        self._emit_progress("render", "Muxing audio onto the stitched video.")
        self._mux_video_audio(silent_video_path, audio_path, muxed_path, tts_clip.duration)

        # 4) Burn subtitles (final encode) or finish without them.
        if ass_subtitles_path and os.path.exists(ass_subtitles_path):
            self._emit_progress(
                "subtitles_burn",
                "Burning subtitles into video via FFmpeg...",
                {"ass_path": ass_subtitles_path},
            )
            self._burn_subtitles_with_ffmpeg(muxed_path, ass_subtitles_path, combined_image_path)
            try:
                os.remove(muxed_path)
            except OSError:
                pass
        else:
            try:
                os.replace(muxed_path, combined_image_path)
            except OSError:
                import shutil
                shutil.copyfile(muxed_path, combined_image_path)

        try:
            tts_clip.close()
        except Exception:
            pass

        success(f'Wrote Video to "{combined_image_path}"')
        self.video_path = os.path.abspath(combined_image_path)
        self._emit_progress(
            "done",
            "Finished writing the final video.",
            {"video_path": combined_image_path},
        )
        self._write_workspace_state()
        return combined_image_path

    def _combine_in_memory(self) -> str:
        """
        Legacy in-memory assembly (composite all scene clips at once). Used when
        youtube_low_memory_render is disabled.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        assets = self._get_scene_aligned_assets()

        if len(assets) == 0:
            raise RuntimeError(
                "No visual assets were generated, so the video cannot be combined."
            )

        combined_image_path = self._get_workspace_path("final_video.mp4")
        threads = get_threads()
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        scene_durations = self._get_scene_durations(max_duration)

        print(colored("[+] Combining visual assets...", "blue"))
        self._emit_progress("combine", "Combining visual assets into the final timeline.")

        clips = []
        for asset, scene_duration in zip(assets, scene_durations):
            clip = self._build_visual_clip(asset, scene_duration)
            clips.append(clip)

        crossfade = get_crossfade_duration()
        if crossfade > 0 and len(clips) > 1:
            for i in range(len(clips)):
                if i > 0:
                    clips[i] = clips[i].fadein(crossfade)
                if i < len(clips) - 1:
                    clips[i] = clips[i].fadeout(crossfade)
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.set_fps(get_video_fps())

        # Generate subtitles and prepare ASS file for FFmpeg burning
        ass_subtitles_path = None
        try:
            subtitles_path = self.generate_subtitles(self.tts_path)
            if self._should_equalize_subtitles():
                equalize_subtitles(subtitles_path, 18)
            subtitle_entries = self._parse_srt_entries_utf8(subtitles_path)
            if not subtitle_entries:
                raise RuntimeError("Subtitle file was empty after generation.")
            # Generate ASS file for FFmpeg subtitle burning (much faster than MoviePy overlay)
            ass_subtitles_path = self._get_workspace_path("subtitles.ass")
            self._generate_ass_subtitles(subtitle_entries, ass_subtitles_path)
            self._emit_progress(
                "subtitles",
                f"Prepared {len(subtitle_entries)} subtitle entries for FFmpeg burning.",
                {"subtitle_entries": len(subtitle_entries), "subtitles_path": ass_subtitles_path},
            )
        except Exception as e:
            warning(f"Failed to generate subtitles, continuing without subtitles: {e}")
            ass_subtitles_path = None

        audio_layers = [tts_clip.set_fps(44100)]

        try:
            random_song = choose_random_song()
            random_song_clip = AudioFileClip(random_song).set_fps(44100)
            random_song_clip = self._apply_audio_ducking(
                tts_clip, random_song_clip, voice_vol=0.15, silence_vol=0.4
            )
            audio_layers.append(random_song_clip)
        except Exception as e:
            warning(f"Failed to add background song, continuing with voiceover only: {e}")

        if get_sound_effects_enabled():
            try:
                sfx_clips = self._build_sound_effects_clips(self.tts_path)
                if sfx_clips:
                    audio_layers.extend(sfx_clips)
                    info(f" => Added {len(sfx_clips)} sound effects to the audio mix.")
            except Exception as e:
                warning(f"Failed to add sound effects: {e}")

        final_audio = CompositeAudioClip(audio_layers)
        final_clip = final_clip.set_audio(final_audio)
        final_clip = final_clip.set_duration(tts_clip.duration)

        # Render video WITHOUT subtitles first (fast)
        # Subtitles will be burned in via FFmpeg in a second pass
        export_codec, export_ffmpeg_params = self._get_video_export_settings()

        if ass_subtitles_path and os.path.exists(ass_subtitles_path):
            # Render to temp file first, then burn subtitles
            temp_video_path = self._get_workspace_path("video_no_subs.mp4")
            self._emit_progress(
                "render",
                f"Writing base video with {export_codec} (subtitles will be burned in next).",
                {"codec": export_codec, "output_path": temp_video_path},
            )
            final_clip.write_videofile(
                temp_video_path,
                threads=threads,
                codec=export_codec,
                audio_codec="aac",
                ffmpeg_params=export_ffmpeg_params,
                logger=self._build_moviepy_render_logger(),
            )

            # Burn subtitles into final video using FFmpeg
            self._emit_progress(
                "subtitles_burn",
                "Burning subtitles into video via FFmpeg...",
                {"ass_path": ass_subtitles_path},
            )
            self._burn_subtitles_with_ffmpeg(temp_video_path, ass_subtitles_path, combined_image_path)

            # Clean up temp file
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
        else:
            # No subtitles - render directly to final path
            self._emit_progress(
                "render",
                f"Writing final video with {export_codec}.",
                {"codec": export_codec, "output_path": combined_image_path},
            )
            final_clip.write_videofile(
                combined_image_path,
                threads=threads,
                codec=export_codec,
                audio_codec="aac",
                ffmpeg_params=export_ffmpeg_params,
                logger=self._build_moviepy_render_logger(),
            )

        success(f'Wrote Video to "{combined_image_path}"')
        self.video_path = os.path.abspath(combined_image_path)
        self._emit_progress(
            "done",
            "Finished writing the final video.",
            {"video_path": combined_image_path},
        )
        self._write_workspace_state()

        return combined_image_path

    def rerender_video_from_workspace(self, workspace_dir: str | None = None) -> str:
        """
        Rebuilds the final video from a previously saved workspace without
        regenerating text, prompts, assets, or TTS.

        Args:
            workspace_dir (str | None): optional workspace path override

        Returns:
            path (str): regenerated MP4 path
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        if not str(getattr(self, "tts_path", "") or "").strip() or not os.path.exists(self.tts_path):
            raise RuntimeError(
                "This workspace does not contain a saved voiceover, so the final video cannot be rebuilt."
            )

        if not self.visual_assets:
            raise RuntimeError(
                "This workspace does not contain saved visual assets, so the final video cannot be rebuilt."
            )

        self._emit_progress(
            "recovery",
            "Loaded saved workspace state. Rebuilding the final video from existing assets.",
            {
                "workspace_dir": self._get_workspace_dir(),
                "asset_count": len(self.visual_assets),
            },
        )
        self._write_workspace_state()
        path = self.combine()

        pricing_report = self._write_pricing_report()
        if isinstance(getattr(self, "metadata", None), dict):
            self.metadata["pricing"] = pricing_report
        self._write_workspace_state()
        return path

    def _ffmpeg_supports_encoder(self, encoder_name: str) -> bool:
        """
        Returns whether the local ffmpeg build exposes a specific encoder.

        Args:
            encoder_name (str): ffmpeg encoder name

        Returns:
            is_supported (bool): True when the encoder is available
        """
        normalized_name = str(encoder_name or "").strip().lower()
        if not normalized_name:
            return False

        cache = getattr(self, "_ffmpeg_encoder_support_cache", {})
        if normalized_name in cache:
            return cache[normalized_name]

        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                check=True,
                capture_output=True,
                text=True,
            )
            is_supported = normalized_name in str(result.stdout or "").lower()
        except (FileNotFoundError, subprocess.CalledProcessError):
            is_supported = False

        cache[normalized_name] = is_supported
        self._ffmpeg_encoder_support_cache = cache
        return is_supported

    def _get_video_export_settings(self) -> tuple[str, list[str]]:
        """
        Chooses the final video encoder settings, preferring NVIDIA NVENC when
        available and falling back to a faster libx264 preset otherwise.

        Returns:
            settings (tuple[str, list[str]]): codec and ffmpeg params
        """
        common_params = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]

        if self._ffmpeg_supports_encoder("h264_nvenc"):
            if get_verbose():
                info(" => Using NVIDIA NVENC for final video export", show_emoji=False)
            return "h264_nvenc", common_params

        if get_verbose():
            info(" => Using libx264 veryfast preset for final video export", show_emoji=False)
        return "libx264", ["-preset", "veryfast", *common_params]

    def _build_moviepy_render_logger(self):
        """
        Returns a MoviePy logger for the active execution environment.

        Returns:
            logger: default terminal logger or GUI-friendly progress logger
        """
        if not callable(self._progress_callback):
            return "bar"

        owner = self

        class GuiRenderLogger(ProgressBarLogger):
            def __init__(self):
                super().__init__(logged_bars=False, min_time_interval=0)

            def bars_callback(self, bar, attr, value, old_value=None):
                if bar != "t":
                    return

                total = float(self.bars.get(bar, {}).get("total") or 0.0)
                current = float(self.bars.get(bar, {}).get("index") or 0.0)
                if total <= 0:
                    return

                progress = max(0.0, min(1.0, current / total))
                owner._emit_progress(
                    "render_progress",
                    f"Rendering final video... {int(progress * 100)}%",
                    {
                        "progress": progress,
                        "current": int(current),
                        "total": int(total),
                    },
                )

        return GuiRenderLogger()

    def _safe_audio_clip_to_soundarray(
        self,
        clip,
        fps: int = 44100,
        quantize: bool = False,
        nbytes: int = 2,
        buffersize: int = 50000,
    ) -> np.ndarray:
        """
        Converts an audio clip into a numpy array without relying on MoviePy's
        generator stacking path that breaks on newer NumPy versions.

        Args:
            clip: MoviePy audio clip
            fps (int): sample rate
            quantize (bool): whether to quantize samples
            nbytes (int): bytes per sample when quantized
            buffersize (int): chunk size used for long clips

        Returns:
            samples (np.ndarray): audio waveform samples
        """
        clip_duration = float(getattr(clip, "duration", 0.0) or 0.0)
        nchannels = int(getattr(clip, "nchannels", 1) or 1)
        stacker = np.vstack if nchannels == 2 else np.hstack
        max_duration = float(buffersize) / float(fps)

        if clip_duration > max_duration:
            chunks = list(
                clip.iter_chunks(
                    fps=fps,
                    quantize=quantize,
                    nbytes=nbytes,
                    chunksize=buffersize,
                )
            )
            if not chunks:
                return np.array([], dtype=np.float32)
            return stacker(chunks)

        return clip.to_soundarray(
            fps=fps,
            quantize=quantize,
            nbytes=nbytes,
            buffersize=buffersize,
        )

    def _apply_audio_ducking(self, voice_clip, music_clip, voice_vol=0.15, silence_vol=0.4):
        """
        Applies audio ducking by pre-rendering the music with a volume envelope
        to a WAV file, then loading it back as an AudioFileClip.
        This avoids MoviePy's per-frame .fl() filter which is extremely slow.

        Args:
            voice_clip: TTS AudioFileClip
            music_clip: Background music AudioFileClip
            voice_vol (float): Music volume multiplier when voice is active
            silence_vol (float): Music volume multiplier during silence

        Returns:
            ducked_music: AudioFileClip with baked-in ducking
        """
        try:
            import soundfile as sf

            fps = 44100
            voice_array = self._safe_audio_clip_to_soundarray(voice_clip, fps=fps)
            if voice_array.size == 0:
                raise RuntimeError("Voice clip did not contain any audio samples.")

            if voice_array.ndim > 1:
                voice_mono = np.mean(voice_array, axis=1)
            else:
                voice_mono = voice_array

            # Compute RMS energy in 50ms windows
            window_size = max(1, int(fps * 0.05))
            total_samples = len(voice_mono)
            rms = np.array([
                np.sqrt(np.mean(voice_mono[start:end] ** 2))
                for start in range(0, total_samples, window_size)
                for end in [min(start + window_size, total_samples)]
            ])

            threshold = np.max(rms) * 0.15 if np.max(rms) > 0 else 0
            is_speech = rms > threshold

            # Build per-sample volume envelope
            volume_envelope = np.ones(total_samples) * silence_vol
            for window_index, speech_present in enumerate(is_speech):
                start = window_index * window_size
                end = min((window_index + 1) * window_size, total_samples)
                if speech_present:
                    volume_envelope[start:end] = voice_vol

            # Smooth the envelope to avoid clicks (100ms fade)
            smooth_window = int(fps * 0.1)
            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                volume_envelope = np.convolve(volume_envelope, kernel, mode="same")

            # Pre-render: apply envelope to music samples and write to WAV
            music_array = self._safe_audio_clip_to_soundarray(music_clip, fps=fps)
            if music_array.size == 0:
                raise RuntimeError("Music clip did not contain any audio samples.")

            # Trim or pad envelope to match music length
            music_len = len(music_array)
            if len(volume_envelope) < music_len:
                volume_envelope = np.pad(volume_envelope, (0, music_len - len(volume_envelope)), constant_values=silence_vol)
            else:
                volume_envelope = volume_envelope[:music_len]

            if music_array.ndim > 1:
                ducked = music_array * volume_envelope.reshape(-1, 1)
            else:
                ducked = music_array * volume_envelope

            ducked_path = self._get_workspace_path("ducked_music.wav")
            sf.write(ducked_path, ducked.astype(np.float32), fps)

            return AudioFileClip(ducked_path).set_fps(fps)
        except Exception as e:
            warning(f"Audio ducking failed, using flat volume: {e}")
            return music_clip.fx(afx.volumex, voice_vol)

    def _build_sound_effects_clips(self, audio_path: str) -> list:
        """
        Scans the voiceover audio with Whisper to get word timestamps,
        then matches trigger words from config to produce timed SFX AudioFileClips.

        Args:
            audio_path (str): Path to the voiceover WAV file

        Returns:
            sfx_clips (list): AudioFileClip instances positioned at trigger timestamps
        """
        sfx_map = get_sound_effects_map()
        if not sfx_map:
            return []

        sfx_volume = get_sound_effects_volume()
        sfx_offset = get_sound_effects_offset()
        sfx_dir = os.path.join(ROOT_DIR, "sfx")
        min_gap = 0.5  # minimum seconds between SFX to avoid overlap

        # Normalize trigger words: strip diacritics and punctuation for matching
        def strip_word(w: str) -> str:
            w = re.sub(r"[\u064B-\u065F\u0670]", "", str(w))  # Arabic diacritics
            return re.sub(r"[.,!?;:،؛؟…\"'()\[\]]", "", w).strip().lower()

        normalized_map = {}
        for trigger, sfx_file in sfx_map.items():
            normalized_map[strip_word(trigger)] = sfx_file

        # Prefer the TTS's own word timings (instant); Whisper only when unavailable.
        words_with_times = [
            {"word": strip_word(word["word"]), "start": word["start"]}
            for word in self._load_tts_word_timings(audio_path)
        ]
        if not words_with_times:
            try:
                segments, _, _, _ = self._transcribe_with_whisper(
                    audio_path, word_timestamps=True,
                )

                for segment in segments:
                    if segment.words:
                        for word in segment.words:
                            words_with_times.append({
                                "word": strip_word(str(word.word)),
                                "start": word.start,
                            })
            except Exception as e:
                warning(f"SFX: Whisper word timestamps failed: {e}")
                return []

        # Match trigger words and build SFX clips
        sfx_clips = []
        last_sfx_time = -999.0

        for wt in words_with_times:
            sfx_file = normalized_map.get(wt["word"])
            if not sfx_file:
                continue

            sfx_path = os.path.join(sfx_dir, sfx_file)
            if not os.path.exists(sfx_path):
                warning(f"SFX file not found: {sfx_path}")
                continue

            trigger_time = max(0.0, wt["start"] + sfx_offset)

            # Skip if too close to previous SFX
            if trigger_time - last_sfx_time < min_gap:
                continue

            try:
                sfx_clip = (
                    AudioFileClip(sfx_path)
                    .set_fps(44100)
                    .fx(afx.volumex, sfx_volume)
                    .set_start(trigger_time)
                )
                sfx_clips.append(sfx_clip)
                last_sfx_time = trigger_time

                if get_verbose():
                    info(f' => SFX: "{wt["word"]}" @ {trigger_time:.2f}s -> {sfx_file}')
            except Exception as e:
                warning(f"SFX: Failed to load {sfx_file}: {e}")

        return sfx_clips

    def _render_video_assets(self, tts_instance: TTS) -> str:
        """
        Generates images, audio, and the final video from the current subject
        and script state.

        Args:
            tts_instance (TTS): Instance of TTS Class.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        self.images = []
        self.visual_assets = []
        self._cost_items = []
        self._cost_notes = []
        self._pixabay_selection_debug = []

        # Generate the Metadata
        if self._manual_metadata_locked and isinstance(getattr(self, "metadata", None), dict) and self.metadata.get("title"):
            self._emit_progress(
                "metadata",
                "Using provided YouTube metadata.",
                {"metadata": self.metadata},
            )
        else:
            self._emit_progress("metadata_start", "Generating metadata...")
            self.generate_metadata()

        # Generate the Image Prompts
        if self._manual_scene_data_locked and getattr(self, "image_prompts", None) and getattr(self, "scene_units", None):
            self._emit_progress(
                "image_prompts",
                f"Using {len(self.image_prompts)} provided scene prompts.",
                {"image_prompts": self.image_prompts},
            )
        else:
            self._emit_progress("prompts_start", "Generating scene image prompts...")
            self.generate_prompts()

        self._emit_progress("assets_start", "Generating visual assets...")
        generated_assets = self._generate_visual_assets()

        if generated_assets == 0:
            raise RuntimeError(
                "Visual asset generation failed for all prompts. Check your AI image quota and Pixabay configuration, then try again."
            )

        if generated_assets < len(self.image_prompts):
            warning(
                f"Generated {generated_assets}/{len(self.image_prompts)} visual assets. Continuing with the successful ones."
            )

        # Generate the TTS
        self._emit_progress("tts_start", "Generating voiceover audio...")
        self.generate_script_to_speech(tts_instance)

        # Combine everything
        self._emit_progress("render_start", "Starting final video render...")
        path = self.combine()

        if get_verbose():
            info(f" => Generated Video: {path}")

        self.video_path = os.path.abspath(path)
        pricing_report = self._write_pricing_report()
        if isinstance(getattr(self, "metadata", None), dict):
            self.metadata["pricing"] = pricing_report

        return path

    def generate_video(self, tts_instance: TTS, seed_topic: str = "") -> str:
        """
        Generates a YouTube Short based on the provided niche and language.

        Args:
            tts_instance (TTS): Instance of TTS Class.
            seed_topic (str): Optional title/keyword to base the video on. When
                provided, it is used as the topic instead of generating one.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        # Generate the Topic (or use the supplied title/keyword)
        self.generate_topic(seed_topic=seed_topic)

        # Generate the Script
        self.generate_script()

        return self._render_video_assets(tts_instance)

    def generate_video_from_existing_script(
        self,
        tts_instance: TTS,
        script: str,
        subject: str = "",
    ) -> str:
        """
        Generates a video while using a caller-provided script.

        Args:
            tts_instance (TTS): Instance of TTS Class.
            script (str): Manual script to narrate
            subject (str): Optional subject/topic override

        Returns:
            path (str): The path to the generated MP4 File.
        """
        self.set_manual_script(script, subject)
        return self._render_video_assets(tts_instance)

    def generate_video_from_editor(
        self,
        tts_instance: TTS,
        script: str,
        subject: str = "",
        metadata: dict | None = None,
        scene_units: list[str] | None = None,
        image_prompts: list[str] | None = None,
    ) -> str:
        """
        Generates a video from caller-edited script, metadata, and scene data.

        Args:
            tts_instance (TTS): TTS provider instance
            script (str): final narration script
            subject (str): subject/topic
            metadata (dict | None): optional metadata override
            scene_units (list[str] | None): optional scene transcript chunks
            image_prompts (list[str] | None): optional scene prompts

        Returns:
            path (str): final generated video path
        """
        self.set_manual_script(script, subject)
        if metadata is not None:
            self.set_manual_metadata(metadata)
        if scene_units is not None or image_prompts is not None:
            self.set_manual_scene_data(scene_units, image_prompts)
        return self._render_video_assets(tts_instance)

    def rebuild_video_from_workspace(
        self,
        tts_instance: TTS | None = None,
        workspace_dir: str | None = None,
        *,
        regenerate_voiceover: bool = False,
        regenerate_assets: bool = False,
    ) -> str:
        """
        Rebuilds a saved workspace after editor changes.

        Args:
            tts_instance (TTS | None): optional TTS instance for fresh voiceover
            workspace_dir (str | None): optional workspace path override
            regenerate_voiceover (bool): whether to synthesize new narration audio
            regenerate_assets (bool): whether to regenerate visual assets

        Returns:
            path (str): rebuilt video path
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        if regenerate_assets:
            self.images = []
            self.visual_assets = []
            self._pixabay_selection_debug = []
            self._emit_progress("assets_start", "Regenerating visual assets from edited scenes...")
            generated_assets = self._generate_visual_assets()
            if generated_assets == 0:
                raise RuntimeError(
                    "Visual asset regeneration failed for all edited scenes."
                )

        if regenerate_voiceover or not str(getattr(self, "tts_path", "") or "").strip() or not os.path.exists(self.tts_path):
            self._emit_progress("tts_start", "Regenerating voiceover audio...")
            self.generate_script_to_speech(tts_instance or TTS())

        if not str(getattr(self, "tts_path", "") or "").strip() or not os.path.exists(self.tts_path):
            raise RuntimeError("This workspace does not contain a saved voiceover, so the video cannot be rebuilt.")

        self._emit_progress("render_start", "Rebuilding the final video from the edited draft...")
        path = self.combine()

        pricing_report = self._write_pricing_report()
        if isinstance(getattr(self, "metadata", None), dict):
            self.metadata["pricing"] = pricing_report
        self._write_workspace_state()
        return path

    def regenerate_scene_asset(
        self,
        scene_index: int,
        provider: str,
        workspace_dir: str | None = None,
    ) -> dict:
        """
        Replaces a single scene asset using either Pixabay or Nano Banana.

        Args:
            scene_index (int): zero-based scene order
            provider (str): pixabay or nanobanana2
            workspace_dir (str | None): optional workspace override

        Returns:
            asset (dict): stored asset payload for the regenerated scene
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        scene_units = list(getattr(self, "scene_units", []) or [])
        image_prompts = list(getattr(self, "image_prompts", []) or [])
        if scene_index < 0 or scene_index >= len(scene_units):
            raise RuntimeError(f"Scene {scene_index + 1} is out of range for this draft.")

        scene_text = str(scene_units[scene_index] or "").strip()
        prompt_text = (
            str(image_prompts[scene_index] or "").strip()
            if scene_index < len(image_prompts)
            else self._make_provider_friendly_image_prompt(scene_text)
        )
        normalized_provider = str(provider or "").strip().lower()
        self._emit_progress(
            "assets_start",
            f"Regenerating asset for scene {scene_index + 1} using {normalized_provider}.",
            {
                "scene_index": scene_index,
                "provider": normalized_provider,
            },
        )

        if normalized_provider == "pixabay":
            preferred_durations = self._estimate_scene_durations_for_asset_search()
            preferred_duration = (
                preferred_durations[scene_index]
                if scene_index < len(preferred_durations)
                else None
            )
            intents, global_context = self._build_scene_stock_intents([scene_text], [prompt_text])
            scene_intent = intents[0] if intents else self._build_fallback_scene_stock_intent(
                scene_index=scene_index,
                scene_text=scene_text,
                prompt_text=prompt_text,
                global_context=global_context,
            )
            candidates = self._collect_pixabay_scene_candidates(
                scene_intent,
                preferred_duration=preferred_duration,
            )
            approved_candidate = next(
                (candidate for candidate in candidates if self._pixabay_candidate_meets_quality_threshold(candidate)),
                None,
            )
            if approved_candidate is None:
                raise RuntimeError(
                    f"No Pixabay asset passed the {PIXBAY_MIN_SELECTION_SCORE:.0f}% quality threshold for scene {scene_index + 1}."
                )

            asset_bytes = self._download_url_bytes(approved_candidate["asset_url"])
            if not asset_bytes:
                raise RuntimeError("Pixabay returned an empty asset payload.")

            if approved_candidate["asset_type"] == "video":
                filename = (
                    f"stock_video_{len(self.visual_assets)+1:02d}_"
                    f"{approved_candidate.get('hit_id') or 'pixabay'}_"
                    f"{approved_candidate.get('asset_variant') or 'clip'}.mp4"
                )
            else:
                filename = f"stock_image_{len(self.visual_assets)+1:02d}_{approved_candidate.get('hit_id') or 'pixabay'}.jpg"

            path = self._persist_binary_asset(
                asset_bytes,
                filename,
                "Pixabay",
                approved_candidate["asset_type"],
                scene_index=scene_index,
            )
            self._record_stock_asset_cost("pixabay", approved_candidate["asset_type"], scene_index=scene_index)
            asset_record = dict(self.visual_assets[-1])
            asset_record.update(
                {
                    "pixabay_hit_id": approved_candidate.get("hit_id"),
                    "pixabay_query": approved_candidate.get("best_query"),
                    "pixabay_query_type": approved_candidate.get("best_query_type"),
                    "pixabay_tags": approved_candidate.get("top_tags", []),
                    "path": path,
                }
            )
            return self._replace_scene_asset_record(scene_index, asset_record)

        if normalized_provider in {"nanobanana2", "nano_banana", "nano banana", "ai"}:
            generated_path = self.generate_image_nanobanana2(prompt_text, scene_index=scene_index)
            if not generated_path:
                raise RuntimeError(f"Nano Banana could not generate a replacement for scene {scene_index + 1}.")
            return self._replace_scene_asset_record(scene_index, dict(self.visual_assets[-1]))

        raise RuntimeError(f'Unsupported scene asset provider "{provider}".')

    def list_pixabay_scene_candidates(
        self,
        scene_index: int,
        workspace_dir: str | None = None,
        limit: int = 8,
    ) -> dict:
        """
        Returns the top Pixabay candidates for one scene without changing the
        stored asset yet.

        Args:
            scene_index (int): zero-based scene order
            workspace_dir (str | None): optional workspace override
            limit (int): max number of candidates to return

        Returns:
            payload (dict): scene context and candidate list
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        scene_units = list(getattr(self, "scene_units", []) or [])
        image_prompts = list(getattr(self, "image_prompts", []) or [])
        if scene_index < 0 or scene_index >= len(scene_units):
            raise RuntimeError(f"Scene {scene_index + 1} is out of range for this draft.")

        scene_text = str(scene_units[scene_index] or "").strip()
        prompt_text = (
            str(image_prompts[scene_index] or "").strip()
            if scene_index < len(image_prompts)
            else self._make_provider_friendly_image_prompt(scene_text)
        )
        preferred_durations = self._estimate_scene_durations_for_asset_search()
        preferred_duration = (
            preferred_durations[scene_index]
            if scene_index < len(preferred_durations)
            else None
        )
        intents, global_context = self._build_scene_stock_intents([scene_text], [prompt_text])
        scene_intent = intents[0] if intents else self._build_fallback_scene_stock_intent(
            scene_index=scene_index,
            scene_text=scene_text,
            prompt_text=prompt_text,
            global_context=global_context,
        )
        ranked_candidates = self._collect_pixabay_scene_candidates(
            scene_intent,
            preferred_duration=preferred_duration,
            max_query_variants=3,
        )
        candidates = [
            candidate
            for candidate in ranked_candidates
            if self._pixabay_candidate_meets_quality_threshold(candidate)
        ][: max(1, int(limit or 8))]
        search_mode = "primary"

        if not candidates:
            widened_intent = dict(scene_intent)
            widened_query_variants = []
            seen_queries = set()

            def add_variant(variant_type: str, query: str) -> None:
                cleaned_query = self._format_pixabay_q_value(
                    re.split(r"\s*\+\s*", str(query or "").strip())
                )
                if not cleaned_query or cleaned_query in seen_queries:
                    return
                seen_queries.add(cleaned_query)
                widened_query_variants.append(
                    {
                        "type": str(variant_type or "variant").strip() or "variant",
                        "query": cleaned_query,
                    }
                )

            for variant in list(scene_intent.get("query_variants", []) or []):
                if isinstance(variant, dict):
                    add_variant(variant.get("type", "variant"), variant.get("query", ""))

            fallback_intent = self._build_fallback_scene_stock_intent(
                scene_index=scene_index,
                scene_text=scene_text,
                prompt_text=prompt_text,
                global_context=global_context,
            )
            for variant in list(fallback_intent.get("query_variants", []) or []):
                if isinstance(variant, dict):
                    add_variant(variant.get("type", "scene_fallback"), variant.get("query", ""))

            add_variant("subject_scene_retry", self.subject)
            add_variant("scene_text_retry", scene_text)
            widened_intent["query_variants"] = widened_query_variants
            widened_intent["query_generation_note"] = (
                str(scene_intent.get("query_generation_note", "") or "").strip()
                + " Widened search retried with fallback scene queries."
            ).strip()

            ranked_candidates = self._collect_pixabay_scene_candidates(
                widened_intent,
                preferred_duration=preferred_duration,
                max_query_variants=5,
            )
            search_mode = "widened_retry"
            candidates = [
                candidate
                for candidate in ranked_candidates
                if self._pixabay_candidate_meets_quality_threshold(candidate)
            ][: max(1, int(limit or 8))]
            if candidates:
                scene_intent = widened_intent
            else:
                scene_intent = widened_intent

        return {
            "scene_index": scene_index,
            "scene_text": scene_text,
            "prompt_text": prompt_text,
            "preferred_duration": preferred_duration,
            "pixabay_query_source": scene_intent.get("pixabay_query_source", "heuristic"),
            "query_generation_note": scene_intent.get("query_generation_note", ""),
            "search_mode": search_mode,
            "query_variants": [
                {
                    "type": str(variant.get("type", "") or "").strip(),
                    "query": str(variant.get("query", "") or "").strip(),
                }
                for variant in list(scene_intent.get("query_variants", []) or [])
                if isinstance(variant, dict)
            ],
            "candidates": [
                {
                    "source": candidate.get("source", "Pixabay"),
                    "asset_type": candidate.get("asset_type"),
                    "asset_url": candidate.get("asset_url"),
                    "asset_variant": candidate.get("asset_variant"),
                    "hit_id": candidate.get("hit_id"),
                    "best_query": candidate.get("best_query"),
                    "best_query_type": candidate.get("best_query_type"),
                    "top_tags": list(candidate.get("top_tags", []) or []),
                    "user_display": candidate.get("user_display"),
                    "selection_score": round(float(self._get_pixabay_selection_score(candidate)), 3),
                    "preview_url": candidate.get("asset_url"),
                    "duration": round(float(candidate.get("duration", 0.0) or 0.0), 3),
                }
                for candidate in candidates
            ],
        }

    def replace_scene_asset_with_pixabay_candidate(
        self,
        scene_index: int,
        candidate_payload: dict,
        workspace_dir: str | None = None,
    ) -> dict:
        """
        Downloads one user-selected Pixabay candidate and links it to a scene.

        Args:
            scene_index (int): zero-based scene order
            candidate_payload (dict): selected Pixabay candidate summary
            workspace_dir (str | None): optional workspace override

        Returns:
            asset (dict): stored asset payload
        """
        if workspace_dir:
            self.load_workspace_state(workspace_dir)

        if not isinstance(candidate_payload, dict):
            raise RuntimeError("A Pixabay candidate must be selected first.")

        asset_url = str(candidate_payload.get("asset_url", "") or "").strip()
        asset_type = str(candidate_payload.get("asset_type", "") or "").strip().lower()
        if asset_type not in {"image", "video"} or not asset_url:
            raise RuntimeError("The selected Pixabay candidate is missing the required asset information.")

        asset_bytes = self._download_url_bytes(asset_url)
        if not asset_bytes:
            raise RuntimeError("Pixabay returned an empty asset payload.")

        if asset_type == "video":
            filename = (
                f"stock_video_{len(self.visual_assets)+1:02d}_"
                f"{candidate_payload.get('hit_id') or 'pixabay'}_"
                f"{candidate_payload.get('asset_variant') or 'clip'}.mp4"
            )
        else:
            filename = f"stock_image_{len(self.visual_assets)+1:02d}_{candidate_payload.get('hit_id') or 'pixabay'}.jpg"

        source_label = str(candidate_payload.get("source") or "Pixabay")
        path = self._persist_binary_asset(
            asset_bytes,
            filename,
            source_label,
            asset_type,
            scene_index=scene_index,
        )
        self._record_stock_asset_cost(source_label.lower(), asset_type, scene_index=scene_index)
        asset_record = dict(self.visual_assets[-1])
        asset_record.update(
            {
                "pixabay_hit_id": candidate_payload.get("hit_id"),
                "pixabay_query": candidate_payload.get("best_query"),
                "pixabay_query_type": candidate_payload.get("best_query_type"),
                "pixabay_tags": list(candidate_payload.get("top_tags", []) or []),
                "path": path,
            }
        )
        return self._replace_scene_asset_record(scene_index, asset_record)

    def get_channel_id(self) -> str:
        """
        Gets the Channel ID of the YouTube Account.

        Returns:
            channel_id (str): The Channel ID.
        """
        if self.browser is None:
            raise RuntimeError(
                "YouTube browser session is not initialized. Recreate this account session with browser access enabled."
            )

        driver = self.browser
        driver.get("https://studio.youtube.com")
        time.sleep(2)
        channel_id = driver.current_url.split("/")[-1]
        self.channel_id = channel_id

        return channel_id

    def _replace_textbox_value(self, element, text: str) -> None:
        """
        Replaces the text content of a YouTube Studio textbox.

        Args:
            element: Selenium web element for the textbox
            text (str): New text to enter

        Returns:
            None
        """
        driver = self.browser
        if driver is None:
            raise RuntimeError("Browser session is not initialized.")

        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});",
            element,
        )
        element.click()
        time.sleep(0.5)
        element.send_keys(Keys.CONTROL, "a")
        time.sleep(0.2)
        element.send_keys(Keys.DELETE)
        time.sleep(0.2)
        element.send_keys(text)

    def _click_element(self, element) -> None:
        """
        Clicks a Selenium element with a JavaScript fallback for Studio dialogs.

        Args:
            element: Selenium element

        Returns:
            None
        """
        driver = self.browser
        if driver is None:
            raise RuntimeError("Browser session is not initialized.")

        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                element,
            )
            time.sleep(0.2)
            element.click()
        except (ElementClickInterceptedException, StaleElementReferenceException):
            driver.execute_script("arguments[0].click();", element)

    def _get_page_text(self) -> str:
        """
        Returns the visible page text in lowercase for lightweight status checks.

        Returns:
            text (str): normalized page text
        """
        driver = self.browser
        if driver is None:
            raise RuntimeError("Browser session is not initialized.")

        try:
            return driver.find_element(By.TAG_NAME, "body").text.strip().lower()
        except NoSuchElementException:
            return ""

    def _find_visible_button_by_text(self, *labels: str):
        """
        Finds the first visible button whose label matches one of the given texts.

        Args:
            labels (str): candidate button labels

        Returns:
            element: matching Selenium element or None
        """
        driver = self.browser
        if driver is None:
            raise RuntimeError("Browser session is not initialized.")

        normalized_labels = {str(label).strip().lower() for label in labels if str(label).strip()}
        if not normalized_labels:
            return None

        # YouTube Studio renders dialog actions as custom elements
        # (ytcp-button / tp-yt-paper-button), not plain <button>, and the
        # visible text can live in a nested node or only in aria-label. Search
        # all of these and read both the rendered text and aria-label.
        buttons = driver.find_elements(
            By.XPATH,
            "//button | //*[@role='button'] | //ytcp-button | //tp-yt-paper-button",
        )

        def button_texts(element) -> list[str]:
            values = []
            for raw in (element.text, element.get_attribute("aria-label")):
                normalized = " ".join(str(raw or "").split()).strip().lower()
                if normalized:
                    values.append(normalized)
            return values

        # Pass 1: exact match (avoids matching the page's own "Publish" button
        # when we are specifically after "Publish anyway").
        for button in buttons:
            try:
                if not button.is_displayed():
                    continue
                if any(text in normalized_labels for text in button_texts(button)):
                    return button
            except StaleElementReferenceException:
                continue

        # Pass 2: substring match for when the element wraps the label in extra
        # text/whitespace (e.g. an icon + "Publish anyway").
        for button in buttons:
            try:
                if not button.is_displayed():
                    continue
                texts = button_texts(button)
                if any(label in text for text in texts for label in normalized_labels):
                    return button
            except StaleElementReferenceException:
                continue

        return None

    def _content_checks_modal_is_open(self) -> bool:
        """
        Returns whether YouTube Studio is showing the content-check warning modal.

        Returns:
            is_open (bool): True when the modal is visible
        """
        page_text = self._get_page_text()
        return (
            "we're still checking your content" in page_text
            or "we are still checking your content" in page_text
            or "still checking your content" in page_text
        )

    def _confirm_publish_through_content_checks(
        self,
        verbose: bool = False,
        timeout_seconds: int = 45,
    ) -> None:
        """
        Drives the final publish past YouTube's "We're still checking your
        content" modal.

        Clicking publish while the automated checks are still running opens a
        confirmation modal with a "Publish anyway" button. The modal can appear
        a few seconds after the click, and clicking once is not always enough
        (the button can be re-rendered), so poll and re-click until the modal
        is gone or the timeout elapses.

        Args:
            verbose (bool): whether to log progress
            timeout_seconds (int): maximum time to spend confirming the publish

        Returns:
            None
        """
        deadline = time.time() + timeout_seconds
        clicked = False

        while time.time() < deadline:
            if not self._content_checks_modal_is_open():
                # Either the modal never appeared (checks already passed and the
                # video published directly) or we already dismissed it.
                if clicked and verbose:
                    info("\t=> Content-check modal cleared; publish confirmed.")
                return

            publish_anyway_button = self._find_visible_button_by_text(
                "Publish anyway", "Publish now"
            )
            if publish_anyway_button is not None:
                if verbose and not clicked:
                    info("\t=> Content-check modal appeared. Clicking 'Publish anyway'.")
                self._click_element(publish_anyway_button)
                clicked = True
                time.sleep(2)
                continue

            # Modal is open but the button is not ready yet; wait and retry.
            time.sleep(1)

        if self._content_checks_modal_is_open() and verbose:
            warning(
                "\t=> Content-check modal was still open after the publish "
                "confirmation timeout."
            )

    def _upload_is_saved_as_private(self, page_text: str | None = None) -> bool:
        """
        Returns whether Studio shows that the uploaded video has been saved.

        Args:
            page_text (str | None): optional cached page text

        Returns:
            is_saved (bool): True when Studio shows "Saved as private"
        """
        normalized_text = str(page_text if page_text is not None else self._get_page_text()).strip().lower()
        return "saved as private" in normalized_text

    def _wait_for_upload_ready(self, timeout_seconds: int = 1800, poll_seconds: float = 5.0) -> None:
        """
        Waits until YouTube Studio confirms the upload has been saved, even if
        the longer content checks are still in progress.

        Args:
            timeout_seconds (int): maximum time to wait
            poll_seconds (float): delay between polls

        Returns:
            None
        """
        driver = self.browser
        if driver is None:
            raise RuntimeError("Browser session is not initialized.")

        verbose = get_verbose()
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            page_text = self._get_page_text()

            if "upload interrupted" in page_text or "resume upload" in page_text:
                raise RuntimeError(
                    "YouTube Studio interrupted the upload and is asking to resume it manually."
                )

            try:
                done_button = driver.find_element(By.ID, YOUTUBE_DONE_BUTTON_ID)
                done_ready = (
                    done_button.is_displayed()
                    and done_button.is_enabled()
                    and str(done_button.get_attribute("aria-disabled") or "").lower() != "true"
                )
            except NoSuchElementException:
                done_ready = False
            except StaleElementReferenceException:
                done_ready = False

            if done_ready and self._upload_is_saved_as_private(page_text):
                return

            if verbose:
                info("\t=> Waiting for YouTube Studio to save the uploaded video before publishing...")
            time.sleep(poll_seconds)

        raise TimeoutException(
            "Timed out while waiting for YouTube Studio to save the uploaded video."
        )

    def upload_video(self) -> bool:
        """
        Uploads the video to YouTube.

        Returns:
            success (bool): Whether the upload was successful or not.
        """
        if self.browser is None:
            raise RuntimeError(
                "YouTube browser session is not initialized. Recreate this account session with browser access enabled."
            )

        try:
            self.get_channel_id()

            driver = self.browser
            verbose = get_verbose()
            wait = WebDriverWait(driver, 30)

            # Go to youtube.com/upload
            driver.get("https://www.youtube.com/upload")

            # Set video file
            FILE_PICKER_TAG = "ytcp-uploads-file-picker"
            file_picker = wait.until(
                EC.presence_of_element_located((By.TAG_NAME, FILE_PICKER_TAG))
            )
            INPUT_TAG = "input"
            file_input = file_picker.find_element(By.TAG_NAME, INPUT_TAG)
            file_input.send_keys(self.video_path)

            # Wait for upload to finish
            time.sleep(5)

            # Set title
            wait.until(lambda drv: len(drv.find_elements(By.ID, YOUTUBE_TEXTBOX_ID)) >= 2)
            textboxes = [
                textbox for textbox in driver.find_elements(By.ID, YOUTUBE_TEXTBOX_ID)
                if textbox.is_displayed()
            ]

            if len(textboxes) < 2:
                raise RuntimeError(
                    f"Could not find the expected YouTube Studio title/description textboxes. Found {len(textboxes)} visible textbox elements."
                )

            title_el = textboxes[0]
            description_el = textboxes[-1]

            if verbose:
                info("\t=> Setting title...")

            self._replace_textbox_value(title_el, self.metadata["title"])

            if verbose:
                info("\t=> Setting description...")

            # Set description
            time.sleep(2)
            self._replace_textbox_value(description_el, self.metadata["description"])

            time.sleep(0.5)

            # Set `made for kids` option
            if verbose:
                info("\t=> Setting `made for kids` option...")

            is_for_kids_checkbox = wait.until(
                EC.presence_of_element_located((By.NAME, YOUTUBE_MADE_FOR_KIDS_NAME))
            )
            is_not_for_kids_checkbox = wait.until(
                EC.presence_of_element_located((By.NAME, YOUTUBE_NOT_MADE_FOR_KIDS_NAME))
            )

            if not self._is_for_kids:
                self._click_element(is_not_for_kids_checkbox)
            else:
                self._click_element(is_for_kids_checkbox)

            time.sleep(0.5)

            # Click next
            if verbose:
                info("\t=> Clicking next...")

            next_button = wait.until(
                EC.element_to_be_clickable((By.ID, YOUTUBE_NEXT_BUTTON_ID))
            )
            next_button.click()

            # Click next again
            if verbose:
                info("\t=> Clicking next again...")
            next_button = wait.until(
                EC.element_to_be_clickable((By.ID, YOUTUBE_NEXT_BUTTON_ID))
            )
            next_button.click()

            # Wait for 2 seconds
            time.sleep(2)

            # Click next again
            if verbose:
                info("\t=> Clicking next again...")
            next_button = wait.until(
                EC.element_to_be_clickable((By.ID, YOUTUBE_NEXT_BUTTON_ID))
            )
            next_button.click()

            # Set visibility to Public (radios are Private[0], Unlisted[1], Public[2])
            if verbose:
                info("\t=> Setting visibility to Public...")

            wait.until(lambda drv: len(drv.find_elements(By.XPATH, YOUTUBE_RADIO_BUTTON_XPATH)) >= 3)
            radio_button = driver.find_elements(By.XPATH, YOUTUBE_RADIO_BUTTON_XPATH)
            self._click_element(radio_button[2])

            if verbose:
                info("\t=> Waiting for YouTube Studio to show that the upload is saved before the final publish step...")
            self._wait_for_upload_ready()

            if verbose:
                info("\t=> Clicking done button...")

            # Click done button
            done_button = wait.until(
                EC.element_to_be_clickable((By.ID, YOUTUBE_DONE_BUTTON_ID))
            )
            self._click_element(done_button)

            # When the automated checks are still running, YouTube opens a
            # "We're still checking your content — Publish anyway?" modal. It
            # can appear a few seconds after the click and may need re-clicking,
            # so confirm the publish through it with polling + retries. If the
            # modal never appears, the click above already published the video.
            self._confirm_publish_through_content_checks(verbose=verbose)

            # Wait for 2 seconds
            time.sleep(2)

            # Get latest video
            if verbose:
                info("\t=> Getting video URL...")

            # Get the latest uploaded video URL
            driver.get(
                f"https://studio.youtube.com/channel/{self.channel_id}/videos/short"
            )
            time.sleep(2)
            videos = driver.find_elements(By.TAG_NAME, "ytcp-video-row")
            first_video = videos[0]
            anchor_tag = first_video.find_element(By.TAG_NAME, "a")
            href = anchor_tag.get_attribute("href")
            if verbose:
                info(f"\t=> Extracting video ID from URL: {href}")
            video_id = href.split("/")[-2]

            # Build URL
            url = build_url(video_id)

            self.uploaded_video_url = url

            if verbose:
                success(f" => Uploaded Video: {url}")

            # Add video to cache
            try:
                self.add_video(
                    {
                        "title": self.metadata["title"],
                        "description": self.metadata["description"],
                        "hashtags": self.metadata.get("hashtags", []),
                        "tags": self.metadata.get("tags", []),
                        "pricing": self.metadata.get("pricing", {}),
                        "url": url,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception as cache_exc:
                warning(f"Uploaded to YouTube, but failed to update the local video cache: {cache_exc}")

            return True
        except Exception as exc:
            if verbose:
                error(f"YouTube upload failed: {exc}")
            raise RuntimeError(f"YouTube upload failed: {exc}") from exc

    def get_videos(self) -> List[dict]:
        """
        Gets the uploaded videos from the YouTube Channel.

        Returns:
            videos (List[dict]): The uploaded videos.
        """
        if not os.path.exists(get_youtube_cache_path()):
            # Create the cache file
            with open(get_youtube_cache_path(), "w", encoding="utf-8") as file:
                json.dump({"videos": []}, file, indent=4, ensure_ascii=False)
            return []

        videos = []
        # Read the cache file
        with open(get_youtube_cache_path(), "r", encoding="utf-8-sig") as file:
            previous_json = json.load(file)
            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    videos = account["videos"]

        return videos
