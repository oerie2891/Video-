"""Context-windowed LLM translation.

Why this exists rather than a call to a machine-translation API per cue:

A subtitle cue is usually half a sentence. "and then he told me" translated in
isolation loses the subject, the tense agreement and the register, and a
sentence split across three cues comes back as three unrelated fragments. So
every request carries a window of cues plus the preceding cues *and their
accepted translations*, and the model is required to return the same number of
units in the same order.

A rolling glossary is accumulated across requests so names, titles and
recurring terminology stay identical from the first cue to the last — the usual
failure of batch translation.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from ..languages import display_name
from ..types import BackendError
from .base import (
    BaseTranslator,
    ProgressFn,
    TextUnit,
    TranslationOptions,
    noop_progress,
)

DEFAULT_ANTHROPIC_MODEL = os.environ.get("SUBTRANS_LLM_MODEL", "claude-sonnet-5-5")
DEFAULT_OPENAI_MODEL = os.environ.get("SUBTRANS_LLM_MODEL", "gpt-4o")

_SUBTITLE_RULES = """\
You are a professional subtitle translator. You are translating a numbered run \
of subtitle cues from a single continuous recording.

Rules:
1. Translate meaning, register and tone — not word for word. The result must \
sound like a native speaker wrote it for screen.
2. Return EXACTLY one translation per input cue, with the same ids, in the same \
order. Never merge, split, drop or reorder cues. A cue that is a sentence \
fragment stays a fragment: translate it so the full sentence reads correctly \
once the cues play in sequence.
3. Keep each translation at or under its char_budget where the language allows \
it. Prefer a shorter natural phrasing over a literal long one — viewers have to \
read this in a few seconds.
4. Do not add explanations, transliterations, notes or honorific padding. Never \
output romanisation alongside the translation.
5. Preserve a leading "-" that marks a speaker change, and preserve the position \
of inline markup such as <i>, {\\an8} and [bracketed sound descriptions] \
(translate the text inside brackets).
6. Keep numerals, units, proper nouns and on-screen text consistent with the \
glossary. Add any new recurring proper noun or technical term to "terms".
7. Never leave a cue empty. If a cue is only filler ("uh", "mm"), give the \
natural equivalent or the closest short interjection.
8. Output nothing but the JSON object.
"""

_DOCUMENT_RULES = """\
You are a professional translator working on a continuous document, supplied as \
numbered blocks.

Rules:
1. Translate faithfully and idiomatically, preserving tone, register and \
paragraph structure.
2. Return EXACTLY one translation per input block, same ids, same order. Never \
merge, split, drop or reorder blocks.
3. Preserve Markdown, list markers, code spans, URLs, numbers and inline \
formatting exactly where they appear. Do not translate code or URLs.
4. Keep terminology consistent with the glossary and add new recurring terms \
to "terms".
5. Add no commentary or translator's notes. Output nothing but the JSON object.
"""


class LLMTranslator(BaseTranslator):
    name = "llm"
    supports_glossary = True

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 4,
        temperature: float = 0.2,
    ) -> None:
        self.provider = (provider or _detect_provider()).lower()
        if self.provider == "anthropic":
            self.model = model or DEFAULT_ANTHROPIC_MODEL
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        else:
            self.model = model or DEFAULT_OPENAI_MODEL
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.max_retries = max_retries
        self.temperature = temperature
        self._client: Any = None

    # -- availability ------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if self.provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                return False, "anthropic package not installed (pip install anthropic)"
            if not self.api_key:
                return False, "ANTHROPIC_API_KEY is not set"
            return True, ""
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai package not installed (pip install openai)"
        if not self.api_key:
            return False, "OPENAI_API_KEY is not set"
        return True, ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        if self.provider == "anthropic":
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        else:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # -- public API --------------------------------------------------------

    def translate(
        self,
        units: list[TextUnit],
        options: TranslationOptions,
        progress: ProgressFn = noop_progress,
    ) -> list[str]:
        if not units:
            return []
        self._get_client()  # fail fast on a missing key

        results: dict[int, str] = {}
        glossary = dict(options.glossary or {})
        window = max(1, options.window)
        total = len(units)
        done = 0

        for start in range(0, total, window):
            batch = units[start:start + window]
            context_units = units[max(0, start - options.context):start]
            context_pairs = [
                (u.text, results.get(u.id, "")) for u in context_units
            ]
            translated, new_terms = self._translate_batch(
                batch, context_pairs, options, glossary
            )
            for unit, text in zip(batch, translated, strict=True):
                results[unit.id] = text
            # Earlier terms win: the first choice is the one viewers already saw.
            for k, v in new_terms.items():
                glossary.setdefault(k, v)
            done += len(batch)
            progress(
                min(0.99, done / total),
                f"Translated {done}/{total} into {display_name(options.target)}",
            )

        options.glossary.update(glossary)
        progress(1.0, f"Translated {total} units into {display_name(options.target)}")
        return [results.get(u.id, u.text) for u in units]

    # -- internals ---------------------------------------------------------

    def _translate_batch(
        self,
        batch: list[TextUnit],
        context_pairs: list[tuple[str, str]],
        options: TranslationOptions,
        glossary: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        prompt = self._build_prompt(batch, context_pairs, options, glossary)
        system = _SUBTITLE_RULES if options.mode == "subtitle" else _DOCUMENT_RULES
        expected = [u.id for u in batch]
        budget = _token_budget(batch)

        last_error: str = ""
        for attempt in range(self.max_retries):
            try:
                raw = self._complete(system, prompt, budget, nudge=attempt > 0)
            except _Retryable as exc:
                last_error = str(exc)
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
                continue
            except Exception as exc:
                raise BackendError(f"Translation request failed: {exc}") from exc

            parsed, terms, problem = _parse_response(raw, expected)
            if parsed is not None:
                return [parsed[i] for i in expected], terms
            last_error = problem
            # Deterministic retry: the model drifted, so stop sampling around.
            self.temperature = 0.0

        # Give up on the batch as a unit and go cue-by-cue so the job still
        # completes. Quality drops here; the caller is told via the warning.
        if len(batch) > 1:
            out: list[str] = []
            for unit in batch:
                single, _ = self._translate_batch([unit], context_pairs, options, glossary)
                out.extend(single)
            return out, {}
        raise BackendError(
            f"Translator could not produce a usable response after "
            f"{self.max_retries} attempts: {last_error}"
        )

    def _build_prompt(
        self,
        batch: list[TextUnit],
        context_pairs: list[tuple[str, str]],
        options: TranslationOptions,
        glossary: dict[str, str],
    ) -> str:
        parts: list[str] = []
        src = display_name(options.source) if options.source else "the source language"
        parts.append(f"Source language: {src}")
        parts.append(f"Target language: {display_name(options.target)} ({options.target})")
        if options.formality:
            parts.append(f"Register: {options.formality}")
        if options.notes:
            parts.append(f"Domain notes: {options.notes}")
        if glossary:
            trimmed = dict(list(glossary.items())[:120])
            parts.append(
                "Glossary (use these exact renderings):\n"
                + json.dumps(trimmed, ensure_ascii=False, indent=0)
            )
        if context_pairs:
            label = "cues" if options.mode == "subtitle" else "blocks"
            lines = [
                f"  {src_text.strip()}  ->  {tgt.strip()}"
                for src_text, tgt in context_pairs if tgt
            ]
            if lines:
                parts.append(
                    f"Preceding {label} already translated (for continuity only — "
                    f"do NOT translate these again):\n" + "\n".join(lines)
                )

        key = "cues" if options.mode == "subtitle" else "blocks"
        payload = {
            key: [
                {
                    "id": u.id,
                    "text": u.text,
                    **({"speaker": u.speaker} if u.speaker else {}),
                    **(
                        {"char_budget": u.max_chars or options.max_chars}
                        if options.mode == "subtitle" and (u.max_chars or options.max_chars)
                        else {}
                    ),
                }
                for u in batch
            ]
        }
        parts.append(
            f"Translate these {len(batch)} {key}:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1)
        )
        parts.append(
            'Reply with JSON only, exactly this shape:\n'
            '{"translations": [{"id": <same id>, "text": "<translation>"}, ...], '
            '"terms": {"<source term>": "<target term>"}}\n'
            f'The "translations" array must contain exactly {len(batch)} items with '
            f'ids {expected_list(batch)}.'
        )
        return "\n\n".join(parts)

    def _complete(self, system: str, prompt: str, max_tokens: int, *, nudge: bool) -> str:
        client = self._get_client()
        if nudge:
            prompt += (
                "\n\nYour previous reply was rejected: it must be valid JSON with "
                "one entry per id, no missing ids, no extra ids, no commentary."
            )
        try:
            if self.provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[
                        {"role": "user", "content": prompt},
                        # Prefill forces the reply to begin as a JSON object.
                        {"role": "assistant", "content": "{"},
                    ],
                )
                body = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                return "{" + body
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if _is_retryable(exc):
                raise _Retryable(str(exc)) from exc
            raise


class _Retryable(RuntimeError):
    pass


def expected_list(batch: list[TextUnit]) -> str:
    ids = [u.id for u in batch]
    if len(ids) <= 8:
        return str(ids)
    return f"[{ids[0]}..{ids[-1]}]"


def _detect_provider() -> str:
    if os.environ.get("SUBTRANS_LLM_PROVIDER"):
        return os.environ["SUBTRANS_LLM_PROVIDER"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


def _token_budget(batch: list[TextUnit]) -> int:
    """Generous ceiling: some targets expand a lot versus the source."""
    chars = sum(len(u.text) for u in batch)
    return max(1024, min(16_000, int(chars * 3) + 600))


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "rate limit", "ratelimit", "429", "overloaded", "529", "timeout",
        "timed out", "503", "502", "500", "connection", "temporarily",
        "apiconnection", "internalserver",
    )
    return any(m in text for m in markers)


def _parse_response(
    raw: str, expected: list[int]
) -> tuple[dict[int, str] | None, dict[str, str], str]:
    """Strict validation. A silently wrong length is worse than a retry."""
    data = _loose_json(raw)
    if data is None:
        return None, {}, "reply was not valid JSON"

    items = None
    for key in ("translations", "cues", "blocks", "results", "items"):
        value = data.get(key)
        if isinstance(value, list):
            items = value
            break
    if items is None:
        return None, {}, "reply had no translations array"

    by_id: dict[int, str] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id", entry.get("index"))
        text = entry.get("text", entry.get("translation"))
        if raw_id is None or text is None:
            continue
        try:
            ident = int(raw_id)
        except (TypeError, ValueError):
            continue
        by_id[ident] = str(text).strip()

    # Positional rescue: right count, wrong/absent ids.
    if set(by_id) != set(expected) and len(items) == len(expected):
        positional: dict[int, str] = {}
        for ident, entry in zip(expected, items, strict=True):
            if isinstance(entry, dict):
                text = entry.get("text", entry.get("translation"))
            else:
                text = entry
            if text is None:
                positional = {}
                break
            positional[ident] = str(text).strip()
        if positional:
            by_id = positional

    missing = [i for i in expected if not by_id.get(i)]
    if missing:
        return None, {}, f"missing or empty translations for ids {missing[:6]}"
    extra = set(by_id) - set(expected)
    for key in extra:
        by_id.pop(key, None)

    terms_raw = data.get("terms") or data.get("glossary") or {}
    terms = {
        str(k): str(v)
        for k, v in (terms_raw.items() if isinstance(terms_raw, dict) else [])
        if k and v and len(str(k)) < 80
    }
    return by_id, terms, ""


def _loose_json(raw: str) -> dict[str, Any] | None:
    """Parse JSON that may be fenced, prefixed, or truncated mid-array."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    # Truncated output: close the structures and salvage complete entries.
    for suffix in ('"}]}', '}]}', ']}', '}'):
        try:
            value = json.loads(text + suffix)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
