import os
import sys
import json
from copy import deepcopy
import srt_equalizer

from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])

DEFAULT_PRICING_CONFIG = {
    "currency": "USD",
    "text_generation": {
        "openai": {
            "models": {
                "gpt-5": {
                    "input_per_1m_tokens": 1.25,
                    "output_per_1m_tokens": 10.00,
                },
                "gpt-5-mini": {
                    "input_per_1m_tokens": 0.25,
                    "output_per_1m_tokens": 2.00,
                },
                "gpt-5-nano": {
                    "input_per_1m_tokens": 0.05,
                    "output_per_1m_tokens": 0.40,
                },
                "gpt-4o-mini": {
                    "input_per_1m_tokens": 0.15,
                    "output_per_1m_tokens": 0.60,
                },
                "default": {
                    "input_per_1m_tokens": 0.00,
                    "output_per_1m_tokens": 0.00,
                },
            }
        },
        "ollama": {
            "models": {
                "default": {
                    "input_per_1m_tokens": 0.00,
                    "output_per_1m_tokens": 0.00,
                }
            }
        },
    },
    "image_generation": {
        "openai": {
            "models": {
                "gpt-image-1": {
                    "qualities": {
                        "low": 0.016,
                        "medium": 0.063,
                        "high": 0.25,
                    }
                },
                "default": {"per_image": 0.00},
            }
        },
        "nanobanana2": {
            "models": {
                "gemini-3.1-flash-image-preview": {"per_image": 0.039},
                "default": {"per_image": 0.039},
            }
        },
        "openrouter": {
            "models": {
                "default": {"per_image": 0.00}
            }
        },
        "qwen_local": {
            "models": {
                # Self-hosted weights: no per-image charge. GPU rental, if any,
                # is billed by the hour and tracked outside this pipeline.
                "default": {"per_image": 0.00}
            }
        },
        "pixabay": {
            "models": {
                "default": {"per_asset": 0.00}
            }
        },
    },
    "tts": {
        "openai": {
            "models": {
                "gpt-4o-mini-tts": {"per_minute_audio": 0.015},
                "default": {"per_minute_audio": 0.015},
            }
        },
        "kitten": {
            "models": {
                "default": {"per_minute_audio": 0.00}
            }
        },
        "edge": {
            "models": {
                "default": {"per_minute_audio": 0.00}
            }
        },
        "deepgram": {
            "models": {
                "default": {"per_1k_characters": 0.030}
            }
        },
    },
    "stt": {
        "script_based": {
            "models": {
                "default": {"per_minute_audio": 0.00}
            }
        },
        "local_whisper": {
            "models": {
                "default": {"per_minute_audio": 0.00}
            }
        },
        "third_party_assemblyai": {
            "models": {
                "default": {"per_minute_audio": 0.0025}
            }
        },
    },
}


def _deep_merge_dicts(base: dict, overrides: dict) -> dict:
    """
    Deep-merges dictionary overrides into a base dictionary.

    Args:
        base (dict): base dictionary
        overrides (dict): user overrides

    Returns:
        merged (dict): merged dictionary
    """
    merged = deepcopy(base)

    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged

def assert_folder_structure() -> None:
    """
    Make sure that the nessecary folder structure is present.

    Returns:
        None
    """
    # Create the .mp folder
    if not os.path.exists(os.path.join(ROOT_DIR, ".mp")):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {os.path.join(ROOT_DIR, '.mp')}", "green"))
        os.makedirs(os.path.join(ROOT_DIR, ".mp"))

def get_first_time_running() -> bool:
    """
    Checks if the program is running for the first time by checking if .mp folder exists.

    Returns:
        exists (bool): True if the program is running for the first time, False otherwise
    """
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))

def get_email_credentials() -> dict:
    """
    Gets the email credentials from the config file.

    Returns:
        credentials (dict): The email credentials
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["email"]

def get_verbose() -> bool:
    """
    Gets the verbose flag from the config file.

    Returns:
        verbose (bool): The verbose flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["verbose"]

def get_firefox_profile_path() -> str:
    """
    Gets the path to the Firefox profile.

    Returns:
        path (str): The path to the Firefox profile
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["firefox_profile"]

def get_headless() -> bool:
    """
    Gets the headless flag from the config file.

    Returns:
        headless (bool): The headless flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["headless"]

def get_ollama_base_url() -> str:
    """
    Gets the Ollama base URL.

    Returns:
        url (str): The Ollama base URL
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("ollama_base_url", "http://127.0.0.1:11434")

def get_llm_provider() -> str:
    """
    Gets the configured LLM provider.

    Returns:
        provider (str): provider name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("llm_provider", "ollama").strip().lower() or "ollama"

def get_ollama_model() -> str:
    """
    Gets the Ollama model name from the config file.

    Returns:
        model (str): The Ollama model name, or empty string if not set.
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("ollama_model", "")

def get_openai_base_url() -> str:
    """
    Gets the OpenAI API base URL.

    Returns:
        url (str): OpenAI API base URL
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_base_url", "https://api.openai.com/v1")

def get_openai_api_key() -> str:
    """
    Gets the OpenAI API key.

    Returns:
        key (str): API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        configured = json.load(file).get("openai_api_key", "")
        return configured or os.environ.get("OPENAI_API_KEY", "")

def get_openai_model() -> str:
    """
    Gets the configured OpenAI model name.

    Returns:
        model (str): OpenAI model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_model", "")

def get_configured_llm_model() -> str:
    """
    Gets the configured model for the active LLM provider.

    Returns:
        model (str): configured model name
    """
    provider = get_llm_provider()
    if provider == "openai":
        return get_openai_model()

    return get_ollama_model()

def get_twitter_language() -> str:
    """
    Gets the Twitter language from the config file.

    Returns:
        language (str): The Twitter language
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["twitter_language"]

def get_twitter_dialect() -> str:
    """
    Gets the Twitter dialect from the config file.

    Returns:
        dialect (str): The Twitter dialect, or an empty string if not set
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("twitter_dialect", "")

def get_nanobanana2_api_base_url() -> str:
    """
    Gets the Nano Banana 2 (Gemini image) API base URL.

    Returns:
        url (str): API base URL
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get(
            "nanobanana2_api_base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        )

def get_image_provider() -> str:
    """
    Gets the configured image generation provider.

    Returns:
        provider (str): image provider name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("image_provider", "nanobanana2").strip().lower() or "nanobanana2"

# Qwen-Image's trained resolutions. Off-grid sizes degrade output noticeably,
# so the provider snaps to these rather than passing arbitrary dimensions.
QWEN_ASPECT_DIMENSIONS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

def get_qwen_dimensions(aspect_ratio: str) -> tuple:
    """
    Maps an aspect ratio to Qwen-Image's nearest trained resolution.

    Args:
        aspect_ratio (str): ratio string such as "9:16"

    Returns:
        dimensions (tuple): (width, height) in pixels
    """
    return QWEN_ASPECT_DIMENSIONS.get(str(aspect_ratio).strip(), QWEN_ASPECT_DIMENSIONS["9:16"])

def get_qwen_api_base_url() -> str:
    """
    Gets the base URL of the self-hosted Qwen-Image server. Point this at
    127.0.0.1 for a local GPU or at a rented box while batch-rendering.

    Returns:
        url (str): base URL without a trailing slash
    """
    value = _read_config().get("qwen_api_base_url", "http://127.0.0.1:8189")
    return str(value).strip().rstrip("/")

def get_qwen_api_key() -> str:
    """
    Gets the shared token for the Qwen-Image server. Optional for loopback,
    but required whenever the server is reachable off this machine.

    Returns:
        key (str): shared token, or "" when unset
    """
    configured = _read_config().get("qwen_api_key", "")
    return str(configured or os.environ.get("QWEN_API_KEY", ""))

def get_qwen_steps() -> int:
    """
    Gets the Qwen-Image sampling step count. This is the main quality/speed
    dial: 50 matches the reference config, 20-25 renders roughly twice as
    fast with a modest quality cost.

    Returns:
        steps (int): number of inference steps
    """
    try:
        return max(1, int(_read_config().get("qwen_steps", 30)))
    except (TypeError, ValueError):
        return 30

def get_qwen_request_timeout() -> int:
    """
    HTTP timeout (seconds) for Qwen-Image requests. A 20B model with CPU
    offload can take many minutes per image, so this is deliberately far
    higher than the hosted-API timeout.

    Returns:
        timeout (int): seconds to wait for one image
    """
    try:
        return max(1, int(_read_config().get("qwen_request_timeout", 1800)))
    except (TypeError, ValueError):
        return 1800

def get_openrouter_api_key() -> str:
    """
    Gets the OpenRouter API key.

    Returns:
        key (str): API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        configured = json.load(file).get("openrouter_api_key", "")
        return configured or os.environ.get("OPENROUTER_API_KEY", "")

def get_openrouter_image_model() -> str:
    """
    Gets the OpenRouter image model name.

    Returns:
        model (str): image model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openrouter_image_model", "black-forest-labs/flux.2-flex")

def get_nanobanana2_api_key() -> str:
    """
    Gets the Nano Banana 2 API key.

    Returns:
        key (str): API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        configured = json.load(file).get("nanobanana2_api_key", "")
        return configured or os.environ.get("GEMINI_API_KEY", "")

def get_nanobanana2_model() -> str:
    """
    Gets the Nano Banana 2 model name.

    Returns:
        model (str): Model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("nanobanana2_model", "gemini-3.1-flash-image-preview")

def get_nanobanana2_aspect_ratio() -> str:
    """
    Gets the aspect ratio for Nano Banana 2 image generation.

    Returns:
        ratio (str): Aspect ratio
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("nanobanana2_aspect_ratio", "9:16")

def get_openai_image_model() -> str:
    """
    Gets the OpenAI image model name.

    Returns:
        model (str): image model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_image_model", "gpt-image-1")

def get_openai_image_quality() -> str:
    """
    Gets the OpenAI image quality setting.

    Returns:
        quality (str): image quality
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_image_quality", "low")

def get_pricing_config() -> dict:
    """
    Gets the pricing configuration used for per-run cost estimates.

    Returns:
        pricing (dict): merged pricing configuration
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        configured = json.load(file).get("pricing", {})

    if not isinstance(configured, dict):
        configured = {}

    return _deep_merge_dicts(DEFAULT_PRICING_CONFIG, configured)

def get_min_image_prompts() -> int:
    """
    Gets the minimum number of image prompts to request for a generated video.

    Returns:
        count (int): minimum prompt count
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("min_image_prompts", 10)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 10

def get_max_image_prompts() -> int:
    """
    Gets the maximum number of image prompts to request for a generated video.

    Returns:
        count (int): maximum prompt count
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("max_image_prompts", 12)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 12

def get_youtube_target_duration_seconds() -> int:
    """
    Gets the target spoken runtime for generated short-form videos.

    Returns:
        seconds (int): target runtime in seconds
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("youtube_target_duration_seconds", 30)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 30

def get_pixabay_api_key() -> str:
    """
    Gets the Pixabay API key.

    Returns:
        key (str): API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        configured = json.load(file).get("pixabay_api_key", "")
        return configured or os.environ.get("PIXABAY_API_KEY", "")

def get_pexels_api_key() -> str:
    """
    Gets the Pexels API key. Pexels is searched alongside Pixabay when set;
    unlike Pixabay it can filter videos to portrait orientation.

    Returns:
        key (str): API key, or "" to skip Pexels
    """
    configured = _read_config().get("pexels_api_key", "")
    return str(configured or os.environ.get("PEXELS_API_KEY", ""))

def get_asset_strategy() -> str:
    """
    Gets the configured media sourcing strategy for generated videos.

    Returns:
        strategy (str): asset strategy
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("asset_strategy", "mixed").strip().lower() or "mixed"

def get_max_ai_assets() -> int:
    """
    Gets the maximum number of AI-generated visual assets to use per video.

    Returns:
        count (int): max AI assets
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("max_ai_assets", 2)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 2

def get_pixabay_results_per_query() -> int:
    """
    Gets the number of Pixabay results to inspect per query.

    Returns:
        count (int): results per query
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("pixabay_results_per_query", 6)
        try:
            return min(20, max(3, int(value)))
        except (TypeError, ValueError):
            return 6

def get_threads() -> int:
    """
    Gets the amount of threads to use for example when writing to a file with MoviePy.

    Returns:
        threads (int): Amount of threads
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["threads"]
    
def get_zip_url() -> str:
    """
    Gets the URL to the zip file containing the songs.

    Returns:
        url (str): The URL to the zip file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["zip_url"]

def get_is_for_kids() -> bool:
    """
    Gets the is for kids flag from the config file.

    Returns:
        is_for_kids (bool): The is for kids flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["is_for_kids"]

def get_google_maps_scraper_zip_url() -> str:
    """
    Gets the URL to the zip file containing the Google Maps scraper.

    Returns:
        url (str): The URL to the zip file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["google_maps_scraper"]

def get_google_maps_scraper_niche() -> str:
    """
    Gets the niche for the Google Maps scraper.

    Returns:
        niche (str): The niche
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["google_maps_scraper_niche"]

def get_scraper_timeout() -> int:
    """
    Gets the timeout for the scraper.

    Returns:
        timeout (int): The timeout
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["scraper_timeout"] or 300

def get_outreach_message_subject() -> str:
    """
    Gets the outreach message subject.

    Returns:
        subject (str): The outreach message subject
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["outreach_message_subject"]
    
def get_outreach_message_body_file() -> str:
    """
    Gets the outreach message body file.

    Returns:
        file (str): The outreach message body file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["outreach_message_body_file"]

def get_tts_voice() -> str:
    """
    Gets the TTS voice from the config file.

    Returns:
        voice (str): The TTS voice
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("tts_voice", "Jasper")

def get_tts_provider() -> str:
    """
    Gets the configured TTS provider.

    Returns:
        provider (str): provider name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("tts_provider", "auto").strip().lower() or "auto"

def get_openai_tts_model() -> str:
    """
    Gets the configured OpenAI TTS model.

    Returns:
        model (str): model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_tts_model", "gpt-4o-mini-tts")

def get_openai_tts_voice() -> str:
    """
    Gets the configured OpenAI TTS voice.

    Returns:
        voice (str): voice name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("openai_tts_voice", "coral")

def get_edge_tts_voice() -> str:
    """
    Gets the Microsoft Edge neural voice used by the free `edge` TTS provider.

    Returns:
        voice (str): voice short name, e.g. ar-EG-ShakirNeural
    """
    return str(_read_config().get("edge_tts_voice", "ar-EG-ShakirNeural")).strip() or "ar-EG-ShakirNeural"

def get_deepgram_api_key() -> str:
    """
    Gets the Deepgram API key used as an English-only TTS fallback.

    Returns:
        key (str): API key, or "" to skip Deepgram
    """
    configured = _read_config().get("deepgram_api_key", "")
    return str(configured or os.environ.get("DEEPGRAM_API_KEY", ""))

def get_deepgram_tts_model() -> str:
    """
    Gets the Deepgram Aura-2 voice model.

    Returns:
        model (str): model name, e.g. aura-2-thalia-en
    """
    return str(_read_config().get("deepgram_tts_model", "aura-2-thalia-en")).strip() or "aura-2-thalia-en"

def get_youtube_metadata_model() -> str:
    """
    Gets the configured lightweight model override for YouTube metadata.

    Returns:
        model (str): model name or empty string
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("youtube_metadata_model", "")

def get_youtube_script_model() -> str:
    """
    Gets the optional model override used only for writing narration scripts
    (e.g. a stronger model for scripts while everything else stays cheap).
    A channel's own "script_model" field takes precedence.

    Returns:
        model (str): model name or empty string
    """
    return str(_read_config().get("youtube_script_model", "") or "").strip()

def get_assemblyai_api_key() -> str:
    """
    Gets the AssemblyAI API key.

    Returns:
        key (str): The AssemblyAI API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["assembly_ai_api_key"]

def get_stt_provider() -> str:
    """
    Gets the configured STT provider.

    Returns:
        provider (str): The STT provider
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("stt_provider", "local_whisper")

def get_whisper_model() -> str:
    """
    Gets the local Whisper model name.

    Returns:
        model (str): Whisper model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("whisper_model", "base")

def get_whisper_device() -> str:
    """
    Gets the target device for Whisper inference.

    Returns:
        device (str): Whisper device
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("whisper_device", "auto")

def get_whisper_compute_type() -> str:
    """
    Gets the compute type for Whisper inference.

    Returns:
        compute_type (str): Whisper compute type
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("whisper_compute_type", "int8")
    
def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    """
    Equalizes the subtitles in a SRT file.

    Args:
        srt_path (str): The path to the SRT file
        max_chars (int): The maximum amount of characters in a subtitle

    Returns:
        None
    """
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
    
def get_font() -> str:
    """
    Gets the font from the config file.

    Returns:
        font (str): The font
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["font"]

def get_subtitle_font() -> str:
    """
    Gets the subtitle font from the config file.

    Returns:
        font (str): subtitle font filename
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_font", get_font())

def get_subtitle_font_english() -> str:
    """
    Gets the subtitle font to use for English/Latin subtitles.

    Returns:
        font (str): english subtitle font filename
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_font_english", get_subtitle_font())

def get_subtitle_font_arabic() -> str:
    """
    Gets the subtitle font to use for Arabic subtitles.

    Returns:
        font (str): arabic subtitle font filename
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_font_arabic", get_subtitle_font())

def get_subtitle_mode() -> str:
    """
    Gets the subtitle timing/render mode.

    Returns:
        mode (str): subtitle mode
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_mode", "word_by_word").strip().lower() or "word_by_word"

def get_subtitle_max_chunk_words() -> int:
    """
    Maximum number of words shown per subtitle in "chunk" mode. Long sentences
    are split into pieces of at most this many words so the screen isn't
    flooded with text. 0 disables the cap (split by sentence only).

    Returns:
        max_words (int): default 5
    """
    value = _read_config().get("subtitle_max_chunk_words", 5)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 5

def get_subtitle_font_size() -> int:
    """
    Gets the subtitle font size.

    Returns:
        size (int): subtitle font size
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("subtitle_font_size", 84)
        try:
            return max(24, int(value))
        except (TypeError, ValueError):
            return 84

def get_subtitle_color() -> str:
    """
    Gets the subtitle text color.

    Returns:
        color (str): subtitle color
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_color", "#FFF7D6")

def get_subtitle_stroke_color() -> str:
    """
    Gets the subtitle stroke color.

    Returns:
        color (str): stroke color
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file).get("subtitle_stroke_color", "#000000")

def get_subtitle_stroke_width() -> int:
    """
    Gets the subtitle stroke width.

    Returns:
        width (int): stroke width
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("subtitle_stroke_width", 6)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 6

def get_fonts_dir() -> str:
    """
    Gets the fonts directory.

    Returns:
        dir (str): The fonts directory
    """
    return os.path.join(ROOT_DIR, "fonts")

def get_imagemagick_path() -> str:
    """
    Gets the path to ImageMagick.

    Returns:
        path (str): The path to ImageMagick
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)["imagemagick_path"]

def get_video_fps() -> int:
    """
    Gets the video frame rate. Default is 30. Use 60 for smoother TikTok videos.

    Returns:
        fps (int): Frames per second (30 or 60)
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("video_fps", 30)
        try:
            fps = int(value)
            return fps if fps in (24, 30, 60) else 30
        except (TypeError, ValueError):
            return 30

def get_crossfade_duration() -> float:
    """
    Gets the crossfade duration in seconds between video scenes.
    0 means no crossfade (hard cut). Default is 0.3 seconds.

    Returns:
        duration (float): Crossfade duration in seconds
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("crossfade_duration", 0.3)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.3

def get_script_sentence_length() -> int:
    """
    Gets the forced script's sentence length.
    In case there is no sentence length in config, returns 4 when none

    Returns:
        length (int): Length of script's sentence
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        config_json = json.load(file)
        if (config_json.get("script_sentence_length") is not None):
            return config_json["script_sentence_length"]
        else:
            return 4

def get_sound_effects_enabled() -> bool:
    """
    Gets whether sound effects are enabled.

    Returns:
        enabled (bool): True if SFX should be added to videos
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return bool(json.load(file).get("sound_effects_enabled", False))

def get_sound_effects_volume() -> float:
    """
    Gets the volume multiplier for sound effects (0.0 to 1.0).

    Returns:
        volume (float): SFX volume multiplier
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("sound_effects_volume", 0.45)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.45

def get_sound_effects_offset() -> float:
    """
    Gets the time offset for SFX placement relative to the trigger word.
    Negative means SFX plays before the word (anticipation).

    Returns:
        offset (float): offset in seconds
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        value = json.load(file).get("sound_effects_offset", -0.15)
        try:
            return max(-1.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return -0.15

def get_sound_effects_map() -> dict:
    """
    Gets the word-to-SFX-file mapping from config.

    Returns:
        mapping (dict): trigger word -> sfx filename
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        mapping = json.load(file).get("sound_effects", {})
        return mapping if isinstance(mapping, dict) else {}

def get_post_bridge_config() -> dict:
    """
    Gets the Post Bridge configuration with safe defaults.

    Returns:
        config (dict): Sanitized Post Bridge configuration
    """
    defaults = {
        "enabled": False,
        "api_key": "",
        "platforms": ["tiktok", "instagram"],
        "account_ids": [],
        "auto_crosspost": False,
    }
    supported_platforms = {"tiktok", "instagram"}

    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        config_json = json.load(file)

    raw_config = config_json.get("post_bridge", {})
    if not isinstance(raw_config, dict):
        raw_config = {}

    raw_platforms = raw_config.get("platforms")
    normalized_platforms = []
    seen_platforms = set()

    if raw_platforms is None:
        normalized_platforms = defaults["platforms"].copy()
    elif isinstance(raw_platforms, list):
        for platform in raw_platforms:
            normalized_platform = str(platform).strip().lower()
            if (
                normalized_platform in supported_platforms
                and normalized_platform not in seen_platforms
            ):
                normalized_platforms.append(normalized_platform)
                seen_platforms.add(normalized_platform)
    else:
        normalized_platforms = []

    raw_account_ids = raw_config.get("account_ids", defaults["account_ids"])
    normalized_account_ids = []
    if isinstance(raw_account_ids, list):
        for account_id in raw_account_ids:
            try:
                normalized_account_ids.append(int(account_id))
            except (TypeError, ValueError):
                continue

    api_key = str(raw_config.get("api_key", "")).strip()
    if not api_key:
        api_key = os.environ.get("POST_BRIDGE_API_KEY", "").strip()

    return {
        "enabled": bool(raw_config.get("enabled", defaults["enabled"])),
        "api_key": api_key,
        "platforms": normalized_platforms,
        "account_ids": normalized_account_ids,
        "auto_crosspost": bool(
            raw_config.get("auto_crosspost", defaults["auto_crosspost"])
        ),
    }

def _read_config() -> dict:
    """
    Reads and returns the raw config.json contents.

    Returns:
        config_json (dict): parsed config, or empty dict on failure
    """
    try:
        with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def get_llm_max_retries() -> int:
    """
    Maximum number of times an LLM call may be retried when it returns
    malformed/empty output. Bounds worst-case token usage on flaky models.

    Returns:
        retries (int): retry cap (>= 0), default 2
    """
    value = _read_config().get("llm_max_retries", 2)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 2

def get_llm_prompt_caching_enabled() -> bool:
    """
    Whether to route static instruction prefixes through the provider's
    prompt-cache path (OpenAI `instructions` + `prompt_cache_key`).

    Returns:
        enabled (bool): default True
    """
    return bool(_read_config().get("llm_prompt_caching_enabled", True))

def get_llm_max_output_tokens() -> dict:
    """
    Per-call output-token caps. Keeps short generations (topic, title, etc.)
    from running long and wasting tokens. Empty/0 means "no cap".

    Returns:
        caps (dict): call-kind -> max output tokens
    """
    # Default to 0 (no cap): reasoning models (gpt-5, o-series) spend output
    # tokens on hidden reasoning, so a low cap can starve the visible answer.
    # Set non-zero caps only for non-reasoning models.
    defaults = {
        "topic": 0,
        "script": 0,
        "metadata": 0,
        "prompts": 0,
        "combined": 0,
        "translate": 0,
        "stock_query": 0,
    }
    raw = _read_config().get("llm_max_output_tokens", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                defaults[str(key)] = max(0, int(value))
            except (TypeError, ValueError):
                continue
    return defaults

def get_image_request_timeout() -> int:
    """
    HTTP timeout (seconds) for image-generation API requests. Lower values
    fail fast when a provider hangs instead of blocking the whole run.

    Returns:
        timeout (int): seconds, default 90
    """
    value = _read_config().get("image_request_timeout", 90)
    try:
        return max(10, int(value))
    except (TypeError, ValueError):
        return 90

def get_youtube_data_api_key() -> str:
    """
    YouTube Data API key used for trend discovery (trending search). Falls back
    to the YOUTUBE_DATA_API_KEY environment variable.

    Returns:
        api_key (str): the key, or "" if not configured
    """
    key = str(_read_config().get("youtube_data_api_key", "") or "").strip()
    if not key:
        key = os.environ.get("YOUTUBE_DATA_API_KEY", "").strip()
    return key

def get_trends_region() -> str:
    """
    Default ISO region code for YouTube trend discovery (e.g. US, EG).

    Returns:
        region (str): region code, default "US"
    """
    return str(_read_config().get("trends_region", "US") or "US").strip()

def get_reddit_client_id() -> str:
    """
    Reddit app client ID for authenticated trend reads (oauth.reddit.com).
    Reddit blocks unauthenticated JSON from many IPs, so this is recommended.
    Falls back to the REDDIT_CLIENT_ID environment variable.

    Returns:
        client_id (str): the id, or "" if not configured
    """
    value = str(_read_config().get("reddit_client_id", "") or "").strip()
    if not value:
        value = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    return value

def get_reddit_client_secret() -> str:
    """
    Reddit app client secret (paired with reddit_client_id). Falls back to the
    REDDIT_CLIENT_SECRET environment variable.

    Returns:
        client_secret (str): the secret, or "" if not configured
    """
    value = str(_read_config().get("reddit_client_secret", "") or "").strip()
    if not value:
        value = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    return value

def get_pixabay_query_use_main_llm() -> bool:
    """
    When True, stock-footage search queries are generated with the configured
    LLM provider (e.g. OpenAI) instead of being forced through Ollama. This
    fixes inaccurate stock queries when Ollama is unavailable.

    Returns:
        enabled (bool): default True
    """
    return bool(_read_config().get("pixabay_query_use_main_llm", True))

def get_openai_reasoning_effort() -> str:
    """
    Reasoning effort for OpenAI reasoning models (gpt-5 / o-series). Lower
    effort means fewer reasoning tokens (cheaper). One of: minimal, low,
    medium, high. Empty string disables sending the parameter.

    Returns:
        effort (str): default "minimal"
    """
    value = str(_read_config().get("openai_reasoning_effort", "minimal")).strip().lower()
    return value if value in {"", "minimal", "low", "medium", "high"} else "minimal"

def get_youtube_combine_metadata_and_prompts() -> bool:
    """
    When True, metadata (title/description/tags) and image prompts are produced
    in a single LLM call instead of two, saving one full-script round trip.

    Returns:
        enabled (bool): default True
    """
    return bool(_read_config().get("youtube_combine_metadata_and_prompts", True))

def get_youtube_send_full_script_to_prompts() -> bool:
    """
    When True, the full script is embedded in the image-prompt call for extra
    context. When False, only the per-scene beats are sent (fewer input tokens).

    Returns:
        enabled (bool): default False
    """
    return bool(_read_config().get("youtube_send_full_script_to_prompts", False))

def get_youtube_template_topic() -> bool:
    """
    When True, the video topic is templated from the niche locally instead of
    calling the LLM to generate it.

    Returns:
        enabled (bool): default False
    """
    return bool(_read_config().get("youtube_template_topic", False))

def get_youtube_low_memory_render() -> bool:
    """
    When True, the final video is assembled by rendering each scene to its own
    file and stitching them with FFmpeg (constant memory), instead of holding
    every scene clip in memory and compositing at once. Prevents out-of-memory
    crashes on long / many-scene videos. Set False to use the legacy in-memory
    path.

    Returns:
        enabled (bool): default True
    """
    return bool(_read_config().get("youtube_low_memory_render", True))

def get_youtube_local_tags() -> bool:
    """
    When True, hashtags/tags are derived locally from the script/subject
    instead of relying solely on the LLM metadata response.

    Returns:
        enabled (bool): default True
    """
    return bool(_read_config().get("youtube_local_tags", True))
