"""Offline translation via NLLB-200 (opt-in: `pip install 'subtrans[local-mt]'`).

Honest assessment: this is the fallback, not the recommended path. NLLB is a
sentence-level model with no cross-sentence context and no notion of a
character budget, so it cannot do what the LLM translator does. What it *can*
do is run with no API key and no network.

To avoid the worst failure mode — translating sentence fragments in isolation —
consecutive units are joined into full sentences, translated whole, then
redistributed across the original units in proportion to source length. That
reconstruction is approximate and degrades for language pairs with very
different word order (English→Japanese being the obvious case).
"""

from __future__ import annotations

import re
from typing import Any

from ..types import BackendError
from .base import (
    BaseTranslator,
    ProgressFn,
    TextUnit,
    TranslationOptions,
    noop_progress,
)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# ISO 639-1 -> FLORES-200 codes used by NLLB.
FLORES: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn", "sk": "slk_Latn",
    "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn", "nn": "nno_Latn",
    "fi": "fin_Latn", "et": "est_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "hu": "hun_Latn", "ro": "ron_Latn", "bg": "bul_Cyrl", "el": "ell_Grek",
    "tr": "tur_Latn", "he": "heb_Hebr", "ar": "arb_Arab", "fa": "pes_Arab",
    "ur": "urd_Arab", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
    "te": "tel_Telu", "ml": "mal_Mlym", "kn": "kan_Knda", "mr": "mar_Deva",
    "gu": "guj_Gujr", "pa": "pan_Guru", "ne": "npi_Deva", "si": "sin_Sinh",
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "jw": "jav_Latn", "su": "sun_Latn", "zh": "zho_Hans",
    "yue": "yue_Hant", "ja": "jpn_Jpan", "ko": "kor_Hang", "my": "mya_Mymr",
    "km": "khm_Khmr", "lo": "lao_Laoo", "ka": "kat_Geor", "hy": "hye_Armn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "tg": "tgk_Cyrl",
    "mn": "khk_Cyrl", "be": "bel_Cyrl", "sr": "srp_Cyrl", "hr": "hrv_Latn",
    "bs": "bos_Latn", "sl": "slv_Latn", "mk": "mkd_Cyrl", "sq": "als_Latn",
    "ca": "cat_Latn", "gl": "glg_Latn", "eu": "eus_Latn", "is": "isl_Latn",
    "ga": "gle_Latn", "cy": "cym_Latn", "mt": "mlt_Latn", "af": "afr_Latn",
    "sw": "swh_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "am": "amh_Ethi",
    "so": "som_Latn", "sn": "sna_Latn", "mg": "plt_Latn", "ht": "hat_Latn",
    "lb": "ltz_Latn", "oc": "oci_Latn", "br": "bre_Latn", "ba": "bak_Cyrl",
    "tt": "tat_Cyrl", "tk": "tuk_Latn", "ps": "pbt_Arab", "sd": "snd_Arab",
    "as": "asm_Beng", "bo": "bod_Tibt", "yi": "ydd_Hebr", "fo": "fao_Latn",
    "la": "ltz_Latn",
}

_TERMINAL = re.compile(r'[.!?…。！？][\s"\'»”’)\]]*$')


class NLLBTranslator(BaseTranslator):
    name = "nllb"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_new_tokens: int = 320,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._pipe: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, (
                "local MT needs torch + transformers "
                "(pip install 'subtrans[local-mt]')"
            )
        return True, ""

    @staticmethod
    def flores(code: str | None) -> str:
        if not code:
            raise BackendError("NLLB requires an explicit source and target language")
        mapped = FLORES.get(code.lower())
        if not mapped:
            raise BackendError(
                f"NLLB has no mapping for language {code!r}. Use the LLM "
                f"translator, which is not limited to this list."
            )
        return mapped

    def _load(self, progress: ProgressFn) -> None:
        if self._model is not None:
            return
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        progress(0.0, f"Loading {self.model_name}… (first run downloads ~2.5 GB)")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self._model.eval()
        self._device = device

    def translate(
        self,
        units: list[TextUnit],
        options: TranslationOptions,
        progress: ProgressFn = noop_progress,
    ) -> list[str]:
        if not units:
            return []
        self._load(progress)
        src = self.flores(options.source)
        tgt = self.flores(options.target)

        groups = _group_sentences(units) if options.mode == "subtitle" else [[u] for u in units]
        sentences = [" ".join(u.text.strip() for u in g).strip() for g in groups]
        translated = self._run(sentences, src, tgt, progress)

        out: dict[int, str] = {}
        for group, whole in zip(groups, translated, strict=True):
            if len(group) == 1:
                out[group[0].id] = whole
            else:
                for unit, piece in zip(group, _redistribute(whole, group), strict=True):
                    out[unit.id] = piece
        return [out.get(u.id, u.text) for u in units]

    def _run(
        self, sentences: list[str], src: str, tgt: str, progress: ProgressFn
    ) -> list[str]:
        import torch

        tokenizer, model = self._tokenizer, self._model
        tokenizer.src_lang = src
        try:
            bos = tokenizer.convert_tokens_to_ids(tgt)
        except Exception:  # pragma: no cover - tokenizer variants
            bos = None

        results: list[str] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start:start + self.batch_size]
            encoded = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self._device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "num_beams": 4,
            }
            if bos is not None:
                kwargs["forced_bos_token_id"] = bos
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            results.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            progress(
                min(0.99, (start + len(chunk)) / max(1, len(sentences))),
                f"Translated {start + len(chunk)}/{len(sentences)} sentences",
            )
        return results


def _group_sentences(units: list[TextUnit]) -> list[list[TextUnit]]:
    """Join consecutive cues into whole sentences before translating."""
    groups: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    for unit in units:
        current.append(unit)
        if _TERMINAL.search(unit.text.strip()):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _redistribute(translation: str, group: list[TextUnit]) -> list[str]:
    """Split a sentence translation back across its cues by source proportion.

    Approximate by construction. Breaks are placed on word boundaries (or
    characters for unspaced scripts) nearest the proportional target.
    """
    lengths = [max(1, len(u.text.strip())) for u in group]
    total = sum(lengths)
    text = translation.strip()
    tokens = text.split(" ")

    if len(tokens) < len(group):  # unspaced script or very short output
        chars = list(text)
        out: list[str] = []
        cursor = 0
        for i, length in enumerate(lengths):
            if i == len(lengths) - 1:
                out.append("".join(chars[cursor:]).strip())
                break
            take = round(len(chars) * length / total)
            out.append("".join(chars[cursor:cursor + take]).strip())
            cursor += take
        return out

    out = []
    cursor = 0
    for i, length in enumerate(lengths):
        if i == len(lengths) - 1:
            out.append(" ".join(tokens[cursor:]).strip())
            break
        take = max(1, round(len(tokens) * length / total))
        take = min(take, len(tokens) - cursor - (len(lengths) - i - 1))
        out.append(" ".join(tokens[cursor:cursor + take]).strip())
        cursor += take
    return out
