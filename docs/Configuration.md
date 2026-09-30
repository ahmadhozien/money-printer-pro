# Configuration

All your configurations will be in a file in the root directory, called `config.json`, which is a copy of `config.example.json`. You can change the values in `config.json` to your liking.

## Values

- `verbose`: `boolean` - If `true`, the application will print out more information.
- `firefox_profile`: `string` - The path to your Firefox profile. This is used to use your Social Media Accounts without having to log in every time you run the application.
- `headless`: `boolean` - If `true`, the application will run in headless mode. This means that the browser will not be visible.
- `llm_provider`: `string` - Text generation provider. Supported values: `ollama`, `openai`.
- `ollama_base_url`: `string` - Base URL of your local Ollama server (default: `http://127.0.0.1:11434`).
- `ollama_model`: `string` - Ollama model to use for text generation (e.g. `llama3.2:3b`). If empty, the app queries Ollama at startup and lets you pick from the available models interactively.
- `openai_base_url`: `string` - Base URL for the OpenAI API (default: `https://api.openai.com/v1`).
- `openai_api_key`: `string` - OpenAI API key. If empty, MPV2 falls back to the `OPENAI_API_KEY` environment variable.
- `openai_model`: `string` - OpenAI model used when `llm_provider` is `openai` (default example: `gpt-5-mini`).
- `image_provider`: `string` - Image generation provider. Supported values: `nanobanana2`, `openai`, `openrouter`, `qwen_local`.
- `asset_strategy`: `string` - Visual sourcing strategy for videos. Supported values: `mixed`, `pixabay_only`, `ai_only`. `mixed` uses Pixabay stock first and only falls back to a limited number of AI-generated visuals. `pixabay_only` always uses the best stock clip found for each scene, even when it scores below the AI-fallback threshold, so clips are not repeated across scenes.
- `max_ai_assets`: `number` - Maximum number of AI-generated visuals allowed per video when `asset_strategy` is `mixed` (default: `2`).
- `openai_image_model`: `string` - OpenAI image model used when `image_provider` is `openai` (default example: `gpt-image-1`).
- `openai_image_quality`: `string` - OpenAI image quality setting (`low`, `medium`, or `high`).
- `openrouter_api_key`: `string` - OpenRouter API key. If empty, MPV2 falls back to the `OPENROUTER_API_KEY` environment variable.
- `openrouter_image_model`: `string` - OpenRouter image model used when `image_provider` is `openrouter` (default example: `black-forest-labs/flux.2-flex`).
- `qwen_api_base_url`: `string` - Base URL of a self-hosted Qwen-Image server (`scripts/qwen_server.py`), used when `image_provider` is `qwen_local` (default: `http://127.0.0.1:8189`). Point it at a rented GPU while batch-rendering. Images cost $0.00 regardless of host.
- `qwen_api_key`: `string` - Shared bearer token for the Qwen-Image server. Optional on loopback, but the server refuses to bind a non-loopback address without one. Falls back to the `QWEN_API_KEY` environment variable.
- `qwen_steps`: `number` - Qwen-Image sampling steps (default: `30`). The main quality/speed dial: `50` matches the reference config, `20`-`25` renders roughly twice as fast.
- `qwen_request_timeout`: `number` - HTTP timeout in seconds for Qwen-Image requests (default: `1800`). Much higher than `image_request_timeout` because a 20B model with CPU offload can take many minutes per image.
- `pixabay_api_key`: `string` - Pixabay API key for free stock image/video lookup. If empty, Pixabay-based strategies will not be able to fetch stock media.
- `pexels_api_key`: `string` - Optional free Pexels API key. When set, Pexels portrait videos/photos are searched alongside Pixabay and ranked by the same scorer (free tier: 200 requests/hour; the run falls back to Pixabay only if it is hit).
- `twitter_language`: `string` - The base language that will be used to generate & post tweets.
- `twitter_dialect`: `string` - Optional dialect/local style for tweets, for example `Egyptian Arabic`, `Gulf Arabic`, or `Moroccan Darija`.
- `nanobanana2_api_base_url`: `string` - Nano Banana 2 API base URL (default: `https://generativelanguage.googleapis.com/v1beta`).
- `nanobanana2_api_key`: `string` - API key for Nano Banana 2 (Gemini image API). If empty, MPV2 falls back to environment variable `GEMINI_API_KEY`.
- `nanobanana2_model`: `string` - Nano Banana 2 model name (default: `gemini-3.1-flash-image-preview`).
- `nanobanana2_aspect_ratio`: `string` - Aspect ratio for generated images (default: `9:16`).
- `youtube_target_duration_seconds`: `number` - Target spoken runtime for generated short-form videos in seconds. This guides script generation (default: `30`). Set `0` to disable runtime targeting.
- `min_image_prompts`: `number` - Minimum number of generated image prompts per video (default: `10`).
- `max_image_prompts`: `number` - Maximum number of generated image prompts per video (default: `12`).
- `pixabay_results_per_query`: `number` - Number of Pixabay search results to inspect per query while sourcing stock visuals (default: `6`).
- `threads`: `number` - The amount of threads that will be used to execute operations, e.g. writing to a file using MoviePy.
- `is_for_kids`: `boolean` - Legacy global fallback for YouTube uploads. Newer builds store this per YouTube account, and this config value is only used when an older account does not yet have its own `is_for_kids` setting.
- `google_maps_scraper`: `string` - The URL to the Google Maps scraper. This will be used to scrape Google Maps for local businesses. It is recommended to use the default value.
- `zip_url`: `string` - The URL to the ZIP file that contains the to be used Songs for the YouTube Shorts Automater.
- `email`: `object`:
    - `smtp_server`: `string` - Your SMTP server.
    - `smtp_port`: `number` - The port of your SMTP server.
    - `username`: `string` - Your email address.
    - `password`: `string` - Your email password.
- `google_maps_scraper_niche`: `string` - The niche you want to scrape Google Maps for.
- `scraper_timeout`: `number` - The timeout for the Google Maps scraper.
- `outreach_message_subject`: `string` - The subject of your outreach message. `{{COMPANY_NAME}}` will be replaced with the company name.
- `outreach_message_body_file`: `string` - The file that contains the body of your outreach message, should be HTML. `{{COMPANY_NAME}}` will be replaced with the company name.
- `stt_provider`: `string` - Provider for subtitle transcription. Default is `local_whisper`. Options:
    * `local_whisper`
    * `third_party_assemblyai`
- `whisper_model`: `string` - Whisper model for local transcription (for example `base`, `small`, `medium`, `large-v3`).
- `whisper_device`: `string` - Device for local Whisper (`auto`, `cpu`, `cuda`).
- `whisper_compute_type`: `string` - Compute type for local Whisper (`int8`, `float16`, etc.).
- `assembly_ai_api_key`: `string` - Your Assembly AI API key. Get yours from [here](https://www.assemblyai.com/app/).
- `tts_provider`: `string` - Text-to-speech provider. Supported values: `auto`, `openai`, `kitten`, `edge`. `auto` prefers OpenAI for Arabic when an OpenAI API key is configured. `edge` uses free Microsoft Edge neural voices (unofficial endpoint) and falls back to Deepgram (English only, if `deepgram_api_key` is set), then OpenAI (Arabic, if `openai_api_key` is set), then KittenTTS.
- `edge_tts_voice`: `string` - Edge voice for `tts_provider: edge` (default: `ar-EG-ShakirNeural`; `ar-EG-SalmaNeural` is the Egyptian female voice; `en-US-GuyNeural` etc. for English).
- `deepgram_api_key`: `string` - Optional Deepgram key used as the English TTS fallback. Deepgram Aura-2 has no Arabic voices.
- `deepgram_tts_model`: `string` - Deepgram Aura-2 voice model (default: `aura-2-thalia-en`).
- `openai_tts_model`: `string` - OpenAI TTS model used when `tts_provider` resolves to OpenAI (default example: `gpt-4o-mini-tts`).
- `openai_tts_voice`: `string` - OpenAI TTS voice used when `tts_provider` resolves to OpenAI (default example: `onyx`).
- `subtitle_font`: `string` - Font file from the `fonts/` folder used for burned-in subtitles (default: `bold_font.ttf`).
- `subtitle_font_size`: `number` - Subtitle font size for burned-in subtitles (default: `84`).
- `subtitle_color`: `string` - Subtitle text color in hex (default: `#FFF7D6`).
- `subtitle_stroke_color`: `string` - Subtitle outline color in hex (default: `#000000`).
- `subtitle_stroke_width`: `number` - Subtitle outline width (default: `6`).
- `youtube_metadata_model`: `string` - Optional cheaper model override for YouTube title/description/hashtags/tags generation. If empty and `llm_provider` is `openai`, MPV2 automatically uses `gpt-5-nano` for this metadata step to save credits.
- `tts_voice`: `string` - Voice for KittenTTS text-to-speech. Default is `Jasper`. Options: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
- `font`: `string` - The font that will be used to generate images. This should be a `.ttf` file in the `fonts/` directory.
- `imagemagick_path`: `string` - The path to the ImageMagick binary. This is used by MoviePy to manipulate images. Install ImageMagick from [here](https://imagemagick.org/script/download.php) and set the path to the `magick.exe` on Windows, or on Linux/MacOS the path to `convert` (usually /usr/bin/convert).
- `script_sentence_length`: `number` - The number of sentences in the generated video script (default: `4`).
- `post_bridge`: `object`:
    - `enabled`: `boolean` - Enables Post Bridge cross-posting after successful YouTube uploads.
    - `api_key`: `string` - Your Post Bridge API key. If empty, MPV2 falls back to `POST_BRIDGE_API_KEY`.
    - `platforms`: `string[]` - Platforms to target. Supported values in v1 are `tiktok` and `instagram`.
    - `account_ids`: `number[]` - Optional fixed Post Bridge account IDs to avoid account-selection prompts.
    - `auto_crosspost`: `boolean` - If `true`, cross-post automatically after a successful YouTube upload. If `false`, interactive runs ask and cron runs skip.

## Example

```json
{
  "verbose": true,
  "firefox_profile": "",
  "headless": false,
  "llm_provider": "ollama",
  "ollama_base_url": "http://127.0.0.1:11434",
  "ollama_model": "",
  "openai_base_url": "https://api.openai.com/v1",
  "openai_api_key": "",
  "openai_model": "gpt-5-mini",
  "image_provider": "nanobanana2",
  "asset_strategy": "mixed",
  "max_ai_assets": 2,
  "openai_image_model": "gpt-image-1",
  "openai_image_quality": "low",
  "openrouter_api_key": "",
  "openrouter_image_model": "black-forest-labs/flux.2-flex",
  "pixabay_api_key": "",
  "pexels_api_key": "",
  "twitter_language": "English",
  "twitter_dialect": "",
  "nanobanana2_api_base_url": "https://generativelanguage.googleapis.com/v1beta",
  "nanobanana2_api_key": "",
  "nanobanana2_model": "gemini-3.1-flash-image-preview",
  "nanobanana2_aspect_ratio": "9:16",
  "youtube_target_duration_seconds": 30,
  "min_image_prompts": 10,
  "max_image_prompts": 12,
  "pixabay_results_per_query": 6,
  "threads": 2,
  "zip_url": "",
  "is_for_kids": false,
  "google_maps_scraper": "https://github.com/gosom/google-maps-scraper/archive/refs/tags/v0.9.7.zip",
  "email": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "",
    "password": ""
  },
  "google_maps_scraper_niche": "",
  "scraper_timeout": 300,
  "outreach_message_subject": "I have a question...",
  "outreach_message_body_file": "outreach_message.html",
  "stt_provider": "local_whisper",
  "whisper_model": "base",
  "whisper_device": "auto",
  "whisper_compute_type": "int8",
  "assembly_ai_api_key": "",
  "tts_provider": "auto",
  "openai_tts_model": "gpt-4o-mini-tts",
  "openai_tts_voice": "onyx",
  "edge_tts_voice": "ar-EG-ShakirNeural",
  "deepgram_api_key": "",
  "deepgram_tts_model": "aura-2-thalia-en",
  "subtitle_font": "bold_font.ttf",
  "subtitle_font_size": 84,
  "subtitle_color": "#FFF7D6",
  "subtitle_stroke_color": "#000000",
  "subtitle_stroke_width": 6,
  "youtube_metadata_model": "",
  "tts_voice": "Jasper",
  "font": "bold_font.ttf",
  "imagemagick_path": "Path to magick.exe or on linux/macOS just /usr/bin/convert",
  "script_sentence_length": 4,
  "post_bridge": {
    "enabled": false,
    "api_key": "",
    "platforms": ["tiktok", "instagram"],
    "account_ids": [],
    "auto_crosspost": false
  }
}
```

## Environment Variable Fallbacks

- `GEMINI_API_KEY`: used when `nanobanana2_api_key` is empty.
- `OPENAI_API_KEY`: used when `openai_api_key` is empty.
- `OPENROUTER_API_KEY`: used when `openrouter_api_key` is empty.
- `PIXABAY_API_KEY`: used when `pixabay_api_key` is empty.
- `PEXELS_API_KEY`: used when `pexels_api_key` is empty.
- `DEEPGRAM_API_KEY`: used when `deepgram_api_key` is empty.
- `POST_BRIDGE_API_KEY`: used when `post_bridge.api_key` is empty.

Example:

```bash
export GEMINI_API_KEY="your_api_key_here"
export POST_BRIDGE_API_KEY="your_post_bridge_api_key_here"
```

See [PostBridge.md](./PostBridge.md) for the full Post Bridge setup and behavior details.

## Token usage controls (LLM cost)

These keys reduce text-LLM token usage during video generation — most relevant when
`llm_provider` is `openai` (paid). All are optional with the defaults shown.

| Key | Default | Effect |
|---|---|---|
| `llm_max_retries` | `2` | Caps retries on malformed/empty LLM output (prevents runaway token usage on flaky models). |
| `llm_prompt_caching_enabled` | `true` | Sends static instruction prefixes via the provider's cache path (`instructions` + `prompt_cache_key` on OpenAI) so repeated prefixes are billed at the cached rate. |
| `openai_reasoning_effort` | `"minimal"` | Reasoning effort for OpenAI reasoning models (gpt-5 / o-series). Lower = fewer reasoning tokens (cheaper). One of `minimal`, `low`, `medium`, `high`, or `""` to omit. |
| `llm_max_output_tokens` | all `0` | Per-call output-token caps (`topic`, `script`, `metadata`, `prompts`, `combined`, `translate`, `stock_query`). **Default 0 (no cap)** — reasoning models spend output tokens on hidden reasoning, so a low cap starves the visible answer. Only set non-zero caps for non-reasoning models. |
| `youtube_low_memory_render` | `true` | Assemble the final video by rendering each scene to a file and stitching with FFmpeg (constant memory), instead of compositing all scene clips in memory at once. Prevents out-of-memory crashes on long / many-scene videos. Set `false` for the legacy in-memory path. |
| `youtube_combine_metadata_and_prompts` | `true` | Generates metadata **and** image prompts in a single LLM call instead of two, eliminating a duplicate full-script round trip. Falls back to two calls automatically if the combined response can't be parsed. |
| `youtube_send_full_script_to_prompts` | `false` | When `false`, only the per-scene beats are sent to the image-prompt call (fewer input tokens). Set `true` to also embed the full script for extra context. |
| `youtube_template_topic` | `false` | When `true`, the topic is templated from the niche locally instead of via an LLM call. |
| `youtube_local_tags` | `true` | Derives tags/hashtags locally from the script when the model omits them, avoiding extra LLM calls. |
| `image_request_timeout` | `90` | HTTP timeout (seconds) for image-generation requests. Lower values fail fast when a provider hangs instead of blocking the run (Nano Banana previously waited 300s). |
| `pixabay_query_use_main_llm` | `true` | Generate stock-footage search queries with the configured LLM provider (e.g. OpenAI) instead of forcing Ollama. Fixes inaccurate stock queries when Ollama is unavailable. |
| `youtube_data_api_key` | `""` | YouTube Data API key for the **Trends** tab (trending search). Reddit trends need no key; YouTube does. Falls back to `YOUTUBE_DATA_API_KEY` env var. Get one free at console.cloud.google.com (enable "YouTube Data API v3"). |
| `trends_region` | `"US"` | Default ISO region code for YouTube trend discovery (e.g. `US`, `EG`, `GB`). |
| `reddit_client_id` / `reddit_client_secret` | `""` | Reddit app credentials for the **Trends** tab. Reddit blocks unauthenticated access from most networks, so these are recommended. Create a free app at reddit.com/prefs/apps (type **script**), then paste the client ID + secret. Env fallbacks: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`. |
| `subtitle_max_chunk_words` | `5` | In `subtitle_mode: "chunk"`, the max words shown per subtitle. Long sentences are split into pieces of at most this many words so the screen isn't flooded with text. `0` disables the cap (split by sentence only). For subtitles that follow the **actual voiceover timing**, set `stt_provider` to `local_whisper` — `script_based` only spreads the script evenly by length and cannot truly sync to the audio. |
