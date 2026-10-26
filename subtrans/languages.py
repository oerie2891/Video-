"""Language metadata: names, codes, script behaviour and honest quality tiers.

Two things here that most tools get wrong and that materially affect output:

1. Line-length and reading-speed limits are *not* universal. CJK subtitles use
   ~16 characters per line, Latin-script ~42. Getting this wrong produces cues
   that are either unreadable or absurdly short.
2. Whisper nominally supports ~99 languages, but word error rate varies by an
   order of magnitude across them. We expose that instead of pretending parity.
"""

from __future__ import annotations

from dataclasses import dataclass

# Whisper's supported transcription languages (ISO 639-1, plus `yue`).
LANGUAGES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen",
    "nn": "Norwegian Nynorsk", "mt": "Maltese", "sa": "Sanskrit",
    "lb": "Luxembourgish", "my": "Burmese", "bo": "Tibetan",
    "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese", "tt": "Tatar",
    "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir",
    "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

NAME_TO_CODE: dict[str, str] = {v.lower(): k for k, v in LANGUAGES.items()}
# A few aliases people actually type.
NAME_TO_CODE.update({
    "mandarin": "zh", "chinese (simplified)": "zh", "chinese (traditional)": "zh",
    "simplified chinese": "zh", "traditional chinese": "zh",
    "brazilian portuguese": "pt", "castilian": "es", "farsi": "fa",
    "burmese (myanmar)": "my", "myanmar": "my", "filipino": "tl",
    "norwegian bokmal": "no", "bokmal": "no", "nynorsk": "nn",
    "flemish": "nl", "greek (modern)": "el", "hebrew (modern)": "he",
})

# Scripts without inter-word spaces: line wrapping must break on characters.
NO_SPACE_LANGUAGES = frozenset({"zh", "yue", "ja", "th", "lo", "my", "km", "bo"})
# Languages conventionally subtitled with tight per-line character limits.
TIGHT_LINE_LANGUAGES = frozenset({"zh", "yue", "ja", "ko"})
RTL_LANGUAGES = frozenset({"ar", "he", "fa", "ur", "ps", "sd", "yi"})

# Transcription quality tiers for Whisper large-v3. Derived from published
# multilingual WER benchmarks; treat as guidance, not a guarantee.
_TIER_GOOD = frozenset({
    "en", "es", "it", "pt", "de", "nl", "fr", "pl", "ca", "ru", "ja", "ko",
    "zh", "sv", "fi", "da", "no", "nn", "cs", "uk", "el", "tr", "hu", "ro",
    "id", "ms", "vi", "he", "ar", "hi", "th", "fa", "ta", "ur", "bg", "sk",
    "hr", "sr", "sl", "lt", "lv", "et", "gl", "af",
})
_TIER_LIMITED = frozenset({
    "mi", "la", "sa", "bo", "haw", "ln", "ba", "tk", "yi", "fo", "lb", "as",
    "tt", "su", "jw", "sn", "oc", "br", "am", "so", "mg", "ha", "yo", "ps",
    "sd", "tg", "my", "km", "lo", "ne", "si",
})


def quality_tier(code: str) -> str:
    """One of 'good', 'fair', 'limited'. Honest about where Whisper degrades."""
    code = normalize(code) or ""
    if code in _TIER_GOOD:
        return "good"
    if code in _TIER_LIMITED:
        return "limited"
    return "fair"


def normalize(value: str | None) -> str | None:
    """Accept a code ('pt'), a name ('Portuguese') or a locale ('pt-BR')."""
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    lowered = v.lower()
    if lowered in ("auto", "detect", "automatic"):
        return None
    if lowered in LANGUAGES:
        return lowered
    # Locale forms: pt-BR, zh_Hans, en_US
    base = lowered.replace("_", "-").split("-")[0]
    if base in LANGUAGES:
        return base
    if lowered in NAME_TO_CODE:
        return NAME_TO_CODE[lowered]
    if base in NAME_TO_CODE:
        return NAME_TO_CODE[base]
    raise ValueError(
        f"Unknown language {value!r}. Use an ISO 639-1 code (e.g. 'es') or an "
        f"English name (e.g. 'Spanish')."
    )


def display_name(code: str | None) -> str:
    if not code:
        return "auto-detected"
    return LANGUAGES.get(code, code)


@dataclass(frozen=True)
class CueStyle:
    """Per-language subtitle presentation constraints.

    Defaults follow widely used broadcast/streaming guidelines: 42 characters
    per line and 17 characters per second for Latin scripts, tighter limits for
    CJK where each glyph carries more information.
    """

    max_chars_per_line: int = 42
    max_lines: int = 2
    max_duration: float = 7.0
    min_duration: float = 0.85
    max_cps: float = 17.0
    min_gap: float = 0.084  # ~2 frames at 24fps; stops cues visually merging
    breaks_on_spaces: bool = True
    rtl: bool = False

    @property
    def max_chars(self) -> int:
        return self.max_chars_per_line * self.max_lines


def style_for(code: str | None) -> CueStyle:
    code = (code or "en").lower()
    if code in TIGHT_LINE_LANGUAGES:
        return CueStyle(
            max_chars_per_line=16,
            max_cps=9.0 if code in ("zh", "yue") else 11.0,
            breaks_on_spaces=code not in NO_SPACE_LANGUAGES,
        )
    if code in NO_SPACE_LANGUAGES:
        return CueStyle(max_chars_per_line=20, max_cps=12.0, breaks_on_spaces=False)
    if code in RTL_LANGUAGES:
        return CueStyle(rtl=True)
    return CueStyle()
