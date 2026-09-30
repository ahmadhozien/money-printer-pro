import io
import os
import re
import numpy as np
import requests
import soundfile as sf
from kittentts import KittenTTS as KittenModel

from config import (
    ROOT_DIR,
    get_deepgram_api_key,
    get_deepgram_tts_model,
    get_edge_tts_voice,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_tts_model,
    get_openai_tts_voice,
    get_tts_provider,
    get_tts_voice,
)
from llm_provider import generate_text
from status import warning

KITTEN_MODEL = "KittenML/kitten-tts-mini-0.8"
KITTEN_SAMPLE_RATE = 24000
DEEPGRAM_MAX_CHARS = 2000  # Deepgram /v1/speak rejects longer inputs

class TTS:
    def __init__(self) -> None:
        self._provider = get_tts_provider()
        self._model = None
        self._voice = get_tts_voice()

    def synthesize(
        self,
        text,
        output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav"),
        language: str | None = None,
        dialect: str | None = None,
        cost_callback=None,
    ):
        text = self._prepare_text_for_tts(text)
        if not text:
            raise ValueError("Cannot synthesize empty text.")

        # Paid providers only run when the free ones before them fail.
        # Deepgram Aura-2 has no Arabic voices, so it only backs up English.
        attempts = []
        if self._provider == "edge":
            attempts.append(("Edge TTS", self._synthesize_edge))
            if get_deepgram_api_key() and not self._is_arabic_narration(text, language, dialect):
                attempts.append(("Deepgram TTS", self._synthesize_deepgram))
        if self._should_use_openai_tts(text, language, dialect):
            attempts.append(("OpenAI TTS", self._synthesize_openai))

        for label, synthesize_with in attempts:
            try:
                return synthesize_with(
                    text,
                    output_file,
                    language,
                    dialect,
                    cost_callback=cost_callback,
                )
            except Exception as provider_error:
                warning(f"{label} failed for this narration, trying the next provider: {provider_error}")

        return self._synthesize_kitten(text, output_file, cost_callback=cost_callback)

    def _synthesize_edge(
        self,
        text: str,
        output_file: str,
        language: str | None = None,
        dialect: str | None = None,
        cost_callback=None,
    ) -> str:
        # ponytail: unofficial free Microsoft endpoint, can break without notice;
        # the provider chain in synthesize() is the safety net.
        import asyncio
        import edge_tts

        voice = get_edge_tts_voice()
        mp3_path = f"{os.path.splitext(output_file)[0]}.edge.mp3"
        asyncio.run(edge_tts.Communicate(text, voice).save(mp3_path))
        audio, sample_rate = sf.read(mp3_path)
        os.remove(mp3_path)
        sf.write(output_file, audio, sample_rate)

        if cost_callback:
            cost_callback(
                {
                    "provider": "edge",
                    "model": "edge-neural",
                    "audio_seconds": len(audio) / float(sample_rate),
                    "characters": len(text),
                    "voice": voice,
                }
            )
        return output_file

    def _synthesize_deepgram(
        self,
        text: str,
        output_file: str,
        language: str | None = None,
        dialect: str | None = None,
        cost_callback=None,
    ) -> str:
        model = get_deepgram_tts_model()
        chunks = []
        for chunk_text in self._split_for_char_limit(text, DEEPGRAM_MAX_CHARS):
            response = requests.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": model, "encoding": "linear16", "container": "wav"},
                headers={"Authorization": f"Token {get_deepgram_api_key()}"},
                json={"text": chunk_text},
                timeout=120,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Deepgram TTS request failed with status {response.status_code}: {response.text.strip()}"
                )
            audio, sample_rate = sf.read(io.BytesIO(response.content))
            chunks.append(audio)

        audio = np.concatenate(chunks)
        sf.write(output_file, audio, sample_rate)

        if cost_callback:
            cost_callback(
                {
                    "provider": "deepgram",
                    "model": model,
                    "audio_seconds": len(audio) / float(sample_rate),
                    "characters": len(text),
                    "voice": model,
                }
            )
        return output_file

    def _split_for_char_limit(self, text: str, max_chars: int) -> list[str]:
        """Packs whole sentences into chunks no longer than max_chars."""
        chunks = [""]
        for sentence in re.split(r"(?<=[.!?؟…])\s+", text):
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if len(chunks[-1]) + len(sentence) + 1 > max_chars:
                chunks.append(sentence)
            else:
                chunks[-1] = f"{chunks[-1]} {sentence}".strip()
        return [chunk for chunk in chunks if chunk]

    def _get_kitten_model(self):
        if self._model is None:
            self._model = KittenModel(KITTEN_MODEL)
        return self._model

    def _synthesize_kitten(self, text: str, output_file: str, cost_callback=None) -> str:
        try:
            audio = self._get_kitten_model().generate(text, voice=self._voice)
        except Exception:
            if not self._contains_arabic(text):
                raise

            warning(
                "KittenTTS failed on Arabic text. Retrying with an English fallback voiceover."
            )
            fallback_text = self._translate_for_tts_fallback(text)
            try:
                audio = self._get_kitten_model().generate(fallback_text, voice=self._voice)
            except Exception as fallback_error:
                raise RuntimeError(
                    "TTS failed for the original Arabic script and the English fallback voiceover."
                ) from fallback_error

        sf.write(output_file, audio, KITTEN_SAMPLE_RATE)
        if cost_callback:
            duration_seconds = max(0.0, float(sf.info(output_file).duration))
            cost_callback(
                {
                    "provider": "kitten",
                    "model": KITTEN_MODEL,
                    "audio_seconds": duration_seconds,
                    "characters": len(text),
                    "voice": self._voice,
                }
            )
        return output_file

    def _should_use_openai_tts(
        self,
        text: str,
        language: str | None = None,
        dialect: str | None = None,
    ) -> bool:
        provider = self._provider
        if provider == "kitten":
            return False

        if not get_openai_api_key():
            return False

        if provider == "openai":
            return True

        return self._is_arabic_narration(text, language, dialect)

    def _is_arabic_narration(
        self,
        text: str,
        language: str | None = None,
        dialect: str | None = None,
    ) -> bool:
        normalized_language = str(language or "").strip().lower()
        normalized_dialect = str(dialect or "").strip().lower()
        return (
            self._contains_arabic(text)
            or "arabic" in normalized_language
            or "darija" in normalized_language
            or "arabic" in normalized_dialect
            or "darija" in normalized_dialect
            or "egyptian" in normalized_dialect
            or "gulf" in normalized_dialect
            or "levantine" in normalized_dialect
        )

    def _build_openai_tts_instructions(
        self,
        text: str,
        language: str | None = None,
        dialect: str | None = None,
    ) -> str:
        normalized_language = str(language or "").strip().lower()
        normalized_dialect = str(dialect or "").strip().lower()
        dialect_hint = str(dialect or "").strip()
        phrasing_suffix = (
            " Read in connected phrases and complete thoughts, not as isolated words. "
            "Keep natural pauses at commas, periods, question marks, and sentence transitions. "
            "Do not spell out letters. Do not pause after every word."
        )

        if "egypt" in normalized_dialect or "egypt" in normalized_language:
            return (
                "Read this as natural spoken Egyptian Arabic. "
                "Speak fluently, conversationally, and with warm natural Egyptian cadence."
                f"{phrasing_suffix}"
            )

        if "gulf" in normalized_dialect or "saudi" in normalized_dialect:
            return (
                "Read this as natural spoken Gulf Arabic. "
                "Speak fluently, conversationally, and with natural Gulf cadence."
                f"{phrasing_suffix}"
            )

        if "levantine" in normalized_dialect:
            return (
                "Read this as natural spoken Levantine Arabic. "
                "Speak fluently, conversationally, and with natural Levantine cadence."
                f"{phrasing_suffix}"
            )

        if "moroccan" in normalized_dialect or "darija" in normalized_dialect:
            return (
                "Read this as natural spoken Moroccan Darija. "
                "Speak fluently, conversationally, and with natural Darija cadence."
                f"{phrasing_suffix}"
            )

        if self._contains_arabic(text) or "arabic" in normalized_language or "darija" in normalized_language:
            return (
                f"Read this as natural spoken Arabic{f' using {dialect_hint}' if dialect_hint else ''}. "
                "Speak fluently and clearly."
                f"{phrasing_suffix}"
            )
        return (
            "Read this naturally and conversationally. "
            "Use sentence-level phrasing and natural pauses, not isolated word-by-word delivery."
        )

    def _prepare_text_for_tts(self, text: str) -> str:
        """
        Normalizes narration text for TTS while preserving useful punctuation
        that helps sentence phrasing and pauses.

        Args:
            text (str): raw narration

        Returns:
            prepared (str): cleaned narration for speech
        """
        prepared = str(text or "").replace("\r", " ").replace("\n", " ")
        prepared = re.sub(r"\s+", " ", prepared).strip()
        prepared = re.sub(r"[\"“”‘’]+", "", prepared)
        prepared = re.sub(r"\s*([،,؛;:.!?؟…])\s*", r"\1 ", prepared)
        prepared = re.sub(r"\s+", " ", prepared).strip()
        return prepared

    def _synthesize_openai(
        self,
        text: str,
        output_file: str,
        language: str | None = None,
        dialect: str | None = None,
        cost_callback=None,
    ) -> str:
        endpoint = f"{get_openai_base_url().rstrip('/')}/audio/speech"
        payload = {
            "model": get_openai_tts_model(),
            "voice": get_openai_tts_voice(),
            "input": text,
            "response_format": "wav",
            "instructions": self._build_openai_tts_instructions(text, language, dialect),
        }
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {get_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            details = response.text.strip()
            if details:
                raise RuntimeError(
                    f"OpenAI TTS request failed with status {response.status_code}: {details}"
                ) from exc
            raise

        with open(output_file, "wb") as audio_file:
            audio_file.write(response.content)

        if cost_callback:
            duration_seconds = max(0.0, float(sf.info(output_file).duration))
            cost_callback(
                {
                    "provider": "openai",
                    "model": get_openai_tts_model(),
                    "audio_seconds": duration_seconds,
                    "characters": len(text),
                    "voice": get_openai_tts_voice(),
                }
            )

        return output_file

    def _contains_arabic(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))

    def _translate_for_tts_fallback(self, text: str) -> str:
        prompt = (
            "Translate the following Arabic video narration into natural spoken English. "
            "Keep the meaning, keep it concise, and return only the translated narration.\n\n"
            f"{text}"
        )
        translated = generate_text(prompt)
        translated = str(translated).strip()
        if not translated:
            raise RuntimeError("Failed to generate an English fallback translation for TTS.")
        return translated
