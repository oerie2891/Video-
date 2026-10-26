"""Turning ASR output into readable subtitle cues.

This is where most of the perceived quality lives. Whisper's native segments
are whatever length the decoder happened to emit — sometimes a single word,
sometimes 30 seconds of run-on text. Shipping those directly as subtitles is
the most common mistake in this category of tool.

The pipeline here is:

    words -> sentence units -> cue split (DP, balanced + linguistic) -> wrap
          -> timing polish (min duration, reading speed, gaps, orphan merge)
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .languages import CueStyle, style_for
from .linebreak import wrap
from .types import Cue, Segment, Subtitles, Transcript, Word

# Don't treat the period in these as a sentence boundary.
_ABBREVIATIONS = frozenset("""mr mrs ms dr prof sr jr st vs etc inc ltd co dept est
    fig no vol approx ca cf eg ie al am pm gen col sgt lt capt rev hon univ
    sra sr dra prof dipl ing bzw usw z.b nr abb""".split())

_SENTENCE_END_RE = re.compile(r'[.!?…。！？]+["\'»”’)\]]*$')
_CLAUSE_END_RE = re.compile(r'[,;:，、；：—–]["\'»”’)\]]*$')
_WORDLIKE_RE = re.compile(r"\w", re.UNICODE)


# --------------------------------------------------------------------------
# Hallucination / noise filtering
# --------------------------------------------------------------------------

def drop_hallucinations(
    transcript: Transcript,
    *,
    no_speech_threshold: float = 0.85,
    logprob_threshold: float = -1.2,
    max_repeats: int = 2,
) -> Transcript:
    """Remove the two failure modes Whisper reliably shows on silence.

    1. Segments the model itself flags as probably-not-speech while also being
       low confidence.
    2. Looped output — the same phrase emitted over and over, which happens on
       music, room tone and long pauses.
    """
    kept: list[Segment] = []
    repeat_run = 0
    previous_norm: str | None = None

    for seg in transcript.segments:
        text = seg.text.strip()
        if not text or not _WORDLIKE_RE.search(text):
            continue
        if (
            seg.no_speech_prob is not None
            and seg.avg_logprob is not None
            and seg.no_speech_prob > no_speech_threshold
            and seg.avg_logprob < logprob_threshold
        ):
            continue

        norm = re.sub(r"\W+", " ", text.lower()).strip()
        if norm and norm == previous_norm:
            repeat_run += 1
            if repeat_run > max_repeats:
                continue
        else:
            repeat_run = 0
        previous_norm = norm

        if _is_internally_looped(text):
            continue
        kept.append(seg)

    return Transcript(
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration=transcript.duration,
        segments=kept,
        source=transcript.source,
        model=transcript.model,
        backend=transcript.backend,
    )


def _is_internally_looped(text: str, min_repeats: int = 4) -> bool:
    """True for 'thanks for watching thanks for watching thanks for watching'."""
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    if len(tokens) < min_repeats * 2:
        return False
    counts = Counter(tokens)
    most_common, hits = counts.most_common(1)[0]
    if hits >= min_repeats and hits / len(tokens) > 0.5 and len(most_common) > 2:
        return True
    # Repeated n-gram covering most of the segment.
    for size in (2, 3, 4):
        if len(tokens) < size * min_repeats:
            continue
        grams = [" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]
        gram, gram_hits = Counter(grams).most_common(1)[0]
        if gram_hits >= min_repeats and gram_hits * size / len(tokens) > 0.6:
            return True
    return False


# --------------------------------------------------------------------------
# Word stream -> sentence units
# --------------------------------------------------------------------------

def _words_from(transcript: Transcript) -> list[Word]:
    """Word stream, synthesising timings when the backend gave none."""
    words: list[Word] = []
    for seg in transcript.segments:
        if seg.words:
            words.extend(w for w in seg.words if w.text.strip())
            continue
        # Proportional fallback: allocate the segment's span by character count.
        tokens = seg.text.split()
        if not tokens:
            continue
        total = sum(len(t) for t in tokens) or 1
        cursor = seg.start
        span = max(seg.duration, 0.001)
        for tok in tokens:
            share = span * (len(tok) / total)
            words.append(Word(start=cursor, end=cursor + share, text=tok))
            cursor += share
    # ASR backends occasionally emit non-monotonic timings; make them sane.
    words.sort(key=lambda w: (w.start, w.end))
    for i, w in enumerate(words):
        if w.end <= w.start:
            nxt = words[i + 1].start if i + 1 < len(words) else w.start + 0.12
            w.end = max(w.start + 0.04, min(w.start + 0.6, nxt))
    return words


def _ends_sentence(token: str) -> bool:
    stripped = token.strip()
    if not _SENTENCE_END_RE.search(stripped):
        return False
    core = stripped.rstrip('.!?…。！？"\'»”’)]').lower()
    if core in _ABBREVIATIONS:
        return False
    if len(core) == 1 and core.isalpha():
        return False  # initials: "J. R. R. Tolkien"
    if re.fullmatch(r"\d+", core):
        return False  # numbered list "1." / decimal fragments
    return True


def _sentence_units(words: list[Word], style: CueStyle, pause: float = 0.65) -> list[list[Word]]:
    units: list[list[Word]] = []
    current: list[Word] = []
    char_cap = style.max_chars * 4

    for i, w in enumerate(words):
        current.append(w)
        text = w.text.strip()
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else math.inf
        chars = sum(len(x.text.strip()) + 1 for x in current)

        if _ends_sentence(text) or gap > pause or chars >= char_cap:
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


# --------------------------------------------------------------------------
# Sentence unit -> cues
# --------------------------------------------------------------------------

def _joined(words: list[Word], spaced: bool) -> str:
    sep = " " if spaced else ""
    return sep.join(w.text.strip() for w in words if w.text.strip()).strip()


def _split_unit(words: list[Word], style: CueStyle) -> list[list[Word]]:
    """Split one sentence into the fewest balanced cues that satisfy limits."""
    spaced = style.breaks_on_spaces
    total_chars = len(_joined(words, spaced))
    total_dur = words[-1].end - words[0].start
    if total_chars <= style.max_chars and total_dur <= style.max_duration:
        return [words]

    n = len(words)
    if n == 1:
        return [words]

    lengths = [len(w.text.strip()) for w in words]
    cum = [0]
    for ln in lengths:
        cum.append(cum[-1] + ln)

    def chars(i: int, j: int) -> int:
        base = cum[j] - cum[i]
        return base + (j - i - 1 if spaced else 0)

    def dur(i: int, j: int) -> float:
        return words[j - 1].end - words[i].start

    def boundary_penalty(j: int) -> float:
        """Cost of ending a cue after words[j-1]."""
        if j >= n:
            return 0.0
        prev = words[j - 1].text.strip()
        gap = words[j].start - words[j - 1].end
        penalty = 0.0
        if _SENTENCE_END_RE.search(prev):
            penalty -= 4.0
        elif _CLAUSE_END_RE.search(prev):
            penalty -= 2.5
        if gap > 0.30:
            penalty -= 2.0      # a real pause is a free cut
        elif gap < 0.04:
            penalty += 1.5      # cutting mid-phrase with no breath
        if prev.endswith("-"):
            penalty += 4.0
        return penalty

    floor = max(
        math.ceil(total_chars / style.max_chars),
        math.ceil(total_dur / style.max_duration),
        1,
    )
    scale = (style.max_chars ** 2) / 110.0

    for parts in range(floor, min(n, floor + 6) + 1):
        dp = [[math.inf] * (n + 1) for _ in range(parts + 1)]
        back = [[-1] * (n + 1) for _ in range(parts + 1)]
        dp[0][0] = 0.0
        for k in range(1, parts + 1):
            for j in range(k, n + 1):
                for i in range(k - 1, j):
                    if dp[k - 1][i] == math.inf:
                        continue
                    c = chars(i, j)
                    if c > style.max_chars or dur(i, j) > style.max_duration:
                        continue
                    cost = dp[k - 1][i] + c * c + boundary_penalty(j) * scale
                    if cost < dp[k][j]:
                        dp[k][j] = cost
                        back[k][j] = i
        if dp[parts][n] < math.inf:
            cuts: list[int] = []
            j, k = n, parts
            while k > 0:
                i = back[k][j]
                cuts.append(i)
                j, k = i, k - 1
            cuts.reverse()
            bounds = cuts + [n]
            return [words[bounds[i]:bounds[i + 1]] for i in range(parts)]

    # Unsatisfiable (e.g. a single very long token): greedy packing by index so
    # we still emit well-formed cues rather than failing.
    out: list[list[Word]] = []
    start = 0
    for end in range(1, n + 1):
        too_wide = chars(start, end) > style.max_chars
        too_long = dur(start, end) > style.max_duration
        if end - start > 1 and (too_wide or too_long):
            out.append(words[start:end - 1])
            start = end - 1
    if start < n:
        out.append(words[start:n])
    return out


def segment_cues(
    transcript: Transcript,
    *,
    language: str | None = None,
    style: CueStyle | None = None,
    filter_hallucinations: bool = True,
) -> Subtitles:
    """Build display-ready cues from a transcript."""
    lang = language or transcript.language
    style = style or style_for(lang)
    if filter_hallucinations:
        transcript = drop_hallucinations(transcript)

    words = _words_from(transcript)
    if not words:
        return Subtitles(cues=[], language=lang, source_language=transcript.language)

    cues: list[Cue] = []
    for unit in _sentence_units(words, style):
        for chunk in _split_unit(unit, style):
            if not chunk:
                continue
            text = _joined(chunk, style.breaks_on_spaces)
            if not text:
                continue
            cues.append(
                Cue(
                    index=len(cues) + 1,
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    lines=wrap(text, style, lang),
                    speaker=None,
                )
            )

    # Carry speaker labels through if the backend provided them.
    _attach_speakers(cues, transcript)
    cues = merge_orphans(cues, style)
    polish_timings(cues, style, media_duration=transcript.duration)
    for i, cue in enumerate(cues, start=1):
        cue.index = i
    return Subtitles(cues=cues, language=lang, source_language=transcript.language)


def _attach_speakers(cues: list[Cue], transcript: Transcript) -> None:
    labelled = [s for s in transcript.segments if s.speaker]
    if not labelled:
        return
    for cue in cues:
        mid = (cue.start + cue.end) / 2
        for seg in labelled:
            if seg.start <= mid <= seg.end:
                cue.speaker = seg.speaker
                break


def merge_orphans(cues: list[Cue], style: CueStyle, max_chars: int = 18) -> list[Cue]:
    """Fold tiny one-or-two-word cues into a neighbour.

    A 0.4s cue reading "Yeah." is visually jarring; merging it where the
    combined text still fits reads far better.
    """
    if len(cues) < 2:
        return cues
    out: list[Cue] = []
    for cue in cues:
        if not out:
            out.append(cue)
            continue
        prev = out[-1]
        gap = cue.start - prev.end
        combined = f"{prev.flat_text} {cue.flat_text}".strip()
        tiny = len(cue.flat_text) <= max_chars or len(prev.flat_text) <= max_chars
        fits = (
            len(combined) <= style.max_chars
            and (cue.end - prev.start) <= style.max_duration
        )
        if tiny and fits and 0 <= gap <= 0.35 and prev.speaker == cue.speaker:
            prev.end = cue.end
            prev.lines = wrap(combined, style)
        else:
            out.append(cue)
    return out


def polish_timings(
    cues: list[Cue],
    style: CueStyle,
    *,
    media_duration: float = 0.0,
) -> None:
    """In-place timing hygiene: no overlaps, readable durations, sane gaps."""
    if not cues:
        return
    limit = media_duration if media_duration > 0 else cues[-1].end + 2.0

    for i, cue in enumerate(cues):
        cue.start = max(0.0, cue.start)
        next_start = cues[i + 1].start if i + 1 < len(cues) else limit
        # Latest end we may claim without colliding with the next cue.
        ceiling = max(cue.start + 0.25, min(limit, next_start - style.min_gap))

        needed_for_reading = len(cue.flat_text) / style.max_cps if style.max_cps > 0 else 0.0
        target = max(cue.end, cue.start + style.min_duration, cue.start + needed_for_reading)
        cue.end = min(max(cue.end, min(target, ceiling)), ceiling)

        if cue.end - cue.start > style.max_duration:
            cue.end = cue.start + style.max_duration

    # Second pass: remove any residual overlap introduced by extension.
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start - style.min_gap:
            cues[i].end = max(
                cues[i].start + 0.25, cues[i + 1].start - style.min_gap
            )


def rewrap(subs: Subtitles, *, language: str | None = None, style: CueStyle | None = None) -> Subtitles:
    """Re-wrap existing cues for a (possibly different) target language.

    Needed after translation: the text length changed, and the target language
    may have entirely different line-length conventions.
    """
    lang = language or subs.language
    style = style or style_for(lang)
    for cue in subs.cues:
        cue.lines = wrap(cue.flat_text, style, lang)
    polish_timings(subs.cues, style)
    subs.language = lang
    return subs


def reading_speed_report(subs: Subtitles, style: CueStyle | None = None) -> dict[str, float | int]:
    """Quality metrics worth showing the user rather than hiding."""
    style = style or style_for(subs.language)
    if not subs.cues:
        return {"cues": 0, "over_cps": 0, "max_cps": 0.0, "mean_cps": 0.0, "over_length": 0}
    cps_values = [c.cps for c in subs.cues if c.duration > 0]
    over_len = sum(
        1 for c in subs.cues
        if any(len(ln) > style.max_chars_per_line for ln in c.lines)
    )
    return {
        "cues": len(subs.cues),
        "over_cps": sum(1 for v in cps_values if v > style.max_cps),
        "max_cps": round(max(cps_values), 1) if cps_values else 0.0,
        "mean_cps": round(sum(cps_values) / len(cps_values), 1) if cps_values else 0.0,
        "over_length": over_len,
    }
