"""Tests for the parts that decide output quality.

Deliberately covers cue segmentation, line breaking, timing hygiene, format
round-trips and translator response validation — the logic that is wrong in
most tools of this kind and that does not need model weights to verify.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from subtrans import languages as L
from subtrans.cues import (
    drop_hallucinations,
    merge_orphans,
    polish_timings,
    reading_speed_report,
    segment_cues,
)
from subtrans.formats import read_subtitles, render, write_srt, write_vtt
from subtrans.languages import CueStyle, style_for
from subtrans.linebreak import wrap
from subtrans.types import Cue, Segment, Subtitles, Transcript, Word

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def words(text: str, start: float = 0.0, wps: float = 2.6) -> list[Word]:
    """Lay a sentence out on a timeline at a plausible speaking rate."""
    out: list[Word] = []
    t = start
    for token in text.split():
        dur = max(0.14, len(token) / (wps * 4.2))
        out.append(Word(start=t, end=t + dur, text=token))
        t += dur + 0.045
    return out


def transcript_of(*sentences: str, gap: float = 0.5) -> Transcript:
    segments: list[Segment] = []
    t = 0.0
    for sentence in sentences:
        ws = words(sentence, start=t)
        segments.append(
            Segment(start=ws[0].start, end=ws[-1].end, text=sentence, words=ws,
                    no_speech_prob=0.01, avg_logprob=-0.2)
        )
        t = ws[-1].end + gap
    return Transcript(language="en", duration=t, segments=segments, model="test")


# --------------------------------------------------------------------------
# languages
# --------------------------------------------------------------------------

def test_normalize_accepts_codes_names_and_locales():
    assert L.normalize("es") == "es"
    assert L.normalize("Spanish") == "es"
    assert L.normalize("pt-BR") == "pt"
    assert L.normalize("zh_Hans") == "zh"
    assert L.normalize("Mandarin") == "zh"
    assert L.normalize("auto") is None
    assert L.normalize("") is None
    assert L.normalize(None) is None


def test_normalize_rejects_nonsense_loudly():
    with pytest.raises(ValueError, match="Unknown language"):
        L.normalize("Klingon")


def test_quality_tiers_are_honest():
    assert L.quality_tier("en") == "good"
    assert L.quality_tier("mi") == "limited"
    assert L.quality_tier("sw") == "fair"


def test_cjk_styles_are_tighter_than_latin():
    assert style_for("ja").max_chars_per_line < style_for("en").max_chars_per_line
    assert style_for("zh").breaks_on_spaces is False
    assert style_for("ko").breaks_on_spaces is True   # Korean does use spaces
    assert style_for("ar").rtl is True


# --------------------------------------------------------------------------
# line breaking
# --------------------------------------------------------------------------

def test_wrap_respects_line_limit():
    style = CueStyle(max_chars_per_line=20, max_lines=2)
    lines = wrap("the quick brown fox jumps over it", style, "en")
    assert len(lines) <= 2
    assert all(len(ln) <= 20 for ln in lines)
    assert " ".join(lines) == "the quick brown fox jumps over it"


def test_wrap_prefers_breaking_after_punctuation():
    style = CueStyle(max_chars_per_line=30, max_lines=2)
    lines = wrap("I finished the migration, and then everything broke", style, "en")
    assert lines[0].endswith(",")


def test_wrap_avoids_orphaning_function_words():
    style = CueStyle(max_chars_per_line=22, max_lines=2)
    lines = wrap("we left the broken server in the cold datacenter", style, "en")
    tail = lines[0].rstrip().split()[-1].lower()
    assert tail not in {"the", "in", "a", "of"}, lines


def test_wrap_single_long_token_does_not_crash():
    style = CueStyle(max_chars_per_line=10, max_lines=2)
    assert wrap("Donaudampfschifffahrtsgesellschaft", style, "de")


def test_wrap_cjk_breaks_on_characters_and_respects_kinsoku():
    style = style_for("ja")
    text = "これはテストです。字幕の改行規則を確認しています。"
    lines = wrap(text, style, "ja")
    assert len(lines) >= 2
    assert all(len(ln) <= style.max_chars_per_line for ln in lines)
    # No line may begin with closing punctuation.
    assert not any(ln[0] in "。、」』）" for ln in lines)


def test_wrap_balances_lines():
    style = CueStyle(max_chars_per_line=40, max_lines=2)
    lines = wrap(
        "this sentence is deliberately long enough to need two full lines here",
        style, "en",
    )
    assert len(lines) == 2
    assert abs(len(lines[0]) - len(lines[1])) < 22, lines


# --------------------------------------------------------------------------
# cue segmentation
# --------------------------------------------------------------------------

def test_segmentation_respects_hard_limits():
    t = transcript_of(
        "This is a fairly long sentence that must be split across more than one "
        "subtitle cue because it simply does not fit inside the character budget.",
        "And here is a second sentence to prove sentences stay separate.",
    )
    subs = segment_cues(t, language="en")
    style = style_for("en")
    assert subs.cues
    for cue in subs.cues:
        assert len(cue.flat_text) <= style.max_chars, cue.flat_text
        assert len(cue.lines) <= style.max_lines
        assert cue.duration <= style.max_duration + 1e-6
        assert all(len(ln) <= style.max_chars_per_line for ln in cue.lines)


def test_cues_never_overlap_and_stay_ordered():
    t = transcript_of(
        "First sentence here.", "Second sentence follows it.",
        "Third sentence closes out the example nicely.",
    )
    subs = segment_cues(t, language="en")
    style = style_for("en")
    for a, b in zip(subs.cues, subs.cues[1:], strict=False):
        assert a.end <= b.start, (a.end, b.start)
        assert b.start - a.end >= style.min_gap - 1e-6
    assert [c.index for c in subs.cues] == list(range(1, len(subs.cues) + 1))


def test_sentence_boundaries_are_not_split_on_abbreviations():
    t = transcript_of("Dr. Smith met Mr. Jones at 3 p.m. for the review.")
    subs = segment_cues(t, language="en")
    # One short sentence: it should not be shattered into many cues.
    assert len(subs.cues) <= 2, [c.flat_text for c in subs.cues]


def test_min_duration_is_enforced_where_room_exists():
    t = Transcript(
        language="en",
        duration=12.0,
        segments=[
            Segment(start=1.0, end=1.3, text="Yes.", words=words("Yes.", 1.0),
                    no_speech_prob=0.0, avg_logprob=-0.1),
            Segment(start=8.0, end=8.6, text="Absolutely not.",
                    words=words("Absolutely not.", 8.0), no_speech_prob=0.0,
                    avg_logprob=-0.1),
        ],
    )
    subs = segment_cues(t, language="en")
    style = style_for("en")
    assert all(c.duration >= style.min_duration - 1e-6 for c in subs.cues)


def test_hallucination_loop_is_dropped():
    looped = Segment(
        start=30.0, end=44.0,
        text="Thanks for watching! Thanks for watching! Thanks for watching! "
             "Thanks for watching! Thanks for watching!",
        words=[], no_speech_prob=0.95, avg_logprob=-1.6,
    )
    real = Segment(
        start=1.0, end=3.0, text="Let's begin the review.",
        words=words("Let's begin the review.", 1.0),
        no_speech_prob=0.02, avg_logprob=-0.3,
    )
    cleaned = drop_hallucinations(Transcript(language="en", duration=45.0,
                                             segments=[real, looped]))
    assert len(cleaned.segments) == 1
    assert "Thanks for watching" not in cleaned.text


def test_low_confidence_silence_is_dropped_but_confident_speech_is_kept():
    noise = Segment(start=0.0, end=2.0, text="♪", words=[],
                    no_speech_prob=0.99, avg_logprob=-2.0)
    quiet_but_real = Segment(start=3.0, end=5.0, text="Barely audible but real.",
                             words=[], no_speech_prob=0.9, avg_logprob=-0.4)
    cleaned = drop_hallucinations(
        Transcript(language="en", duration=6.0, segments=[noise, quiet_but_real])
    )
    assert [s.text for s in cleaned.segments] == ["Barely audible but real."]


def test_orphan_cues_are_merged():
    style = style_for("en")
    cues = [
        Cue(index=1, start=0.0, end=1.0, lines=["I think so."]),
        Cue(index=2, start=1.1, end=1.5, lines=["Yes."]),
    ]
    merged = merge_orphans(cues, style)
    assert len(merged) == 1
    assert merged[0].flat_text == "I think so. Yes."


def test_timing_polish_does_not_steal_the_next_cue_start():
    style = style_for("en")
    cues = [
        Cue(index=1, start=0.0, end=0.4, lines=["A very long line of text indeed"]),
        Cue(index=2, start=1.0, end=3.0, lines=["Next"]),
    ]
    polish_timings(cues, style, media_duration=10.0)
    assert cues[0].end <= cues[1].start - style.min_gap + 1e-9


def test_segmentation_works_without_word_timestamps():
    seg = Segment(
        start=0.0, end=6.0,
        text="No word timings here, so the layer must synthesise them from "
             "character counts instead.",
        words=[], no_speech_prob=0.0, avg_logprob=-0.2,
    )
    subs = segment_cues(Transcript(language="en", duration=6.0, segments=[seg]))
    assert subs.cues
    assert all(c.end > c.start for c in subs.cues)
    for a, b in zip(subs.cues, subs.cues[1:], strict=False):
        assert a.end <= b.start


def test_reading_speed_report_flags_fast_cues():
    subs = Subtitles(
        cues=[Cue(index=1, start=0.0, end=0.5, lines=["A" * 60])], language="en"
    )
    report = reading_speed_report(subs)
    assert report["over_cps"] == 1
    assert report["max_cps"] > 17


# --------------------------------------------------------------------------
# formats
# --------------------------------------------------------------------------

def test_srt_timestamp_format_and_indices():
    subs = Subtitles(
        cues=[
            Cue(index=1, start=0.0, end=1.5, lines=["First"]),
            Cue(index=2, start=3661.25, end=3663.0, lines=["Second", "line"]),
        ],
        language="en",
    )
    out = write_srt(subs)
    assert "00:00:00,000 --> 00:00:01,500" in out
    assert "01:01:01,250 --> 01:01:03,000" in out
    assert out.startswith("1\n")
    assert "Second\nline" in out


def test_srt_round_trip_preserves_timings_and_text(tmp_path: Path):
    original = Subtitles(
        cues=[
            Cue(index=1, start=1.234, end=3.456, lines=["Hello there,", "general."]),
            Cue(index=2, start=4.0, end=6.5, lines=["Second cue."]),
        ],
        language="en",
    )
    path = tmp_path / "a.srt"
    path.write_text(write_srt(original), encoding="utf-8")
    parsed = read_subtitles(path)
    assert len(parsed.cues) == 2
    assert parsed.cues[0].start == pytest.approx(1.234, abs=0.001)
    assert parsed.cues[0].end == pytest.approx(3.456, abs=0.001)
    assert parsed.cues[0].lines == ["Hello there,", "general."]
    assert parsed.cues[1].flat_text == "Second cue."


def test_vtt_round_trip_with_voice_tags(tmp_path: Path):
    subs = Subtitles(
        cues=[Cue(index=1, start=0.5, end=2.0, lines=["Hi <there> & welcome"],
                  speaker="Alex")],
        language="en",
    )
    path = tmp_path / "a.vtt"
    path.write_text(write_vtt(subs), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "00:00:00.500 --> 00:00:02.000" in text
    parsed = read_subtitles(path)
    assert parsed.cues[0].speaker == "Alex"
    assert parsed.cues[0].flat_text == "Hi <there> & welcome"


def test_parse_handles_crlf_bom_and_missing_indices(tmp_path: Path):
    raw = (
        "﻿00:00:01,000 --> 00:00:02,000\r\nNo index line\r\n\r\n"
        "2\r\n00:00:03,000 --> 00:00:04,000\r\nSecond\r\n"
    )
    path = tmp_path / "messy.srt"
    path.write_bytes(raw.encode("utf-8"))
    parsed = read_subtitles(path)
    assert [c.flat_text for c in parsed.cues] == ["No index line", "Second"]


def test_ass_round_trip(tmp_path: Path):
    subs = Subtitles(
        cues=[Cue(index=1, start=1.0, end=2.5, lines=["Line one", "Line two"])],
        language="en",
    )
    path = tmp_path / "a.ass"
    path.write_text(render(subs, "ass"), encoding="utf-8")
    parsed = read_subtitles(path)
    assert parsed.cues[0].lines == ["Line one", "Line two"]
    assert parsed.cues[0].start == pytest.approx(1.0, abs=0.02)


def test_txt_output_builds_paragraphs_not_cue_soup():
    subs = Subtitles(
        cues=[
            Cue(index=1, start=0.0, end=1.0, lines=["One."]),
            Cue(index=2, start=1.05, end=2.0, lines=["Two."]),
            Cue(index=3, start=9.0, end=10.0, lines=["Much later."]),
        ],
        language="en",
    )
    out = render(subs, "txt", timestamps=False)
    assert "One. Two." in out
    assert out.count("\n\n") == 1  # exactly one paragraph break


def test_json_output_is_valid_and_complete():
    subs = Subtitles(
        cues=[Cue(index=1, start=0.0, end=1.0, lines=["Hi"], source_text="Hola")],
        language="en", source_language="es",
    )
    data = json.loads(render(subs, "json"))
    assert data["language"] == "en"
    assert data["source_language"] == "es"
    assert data["cues"][0]["text"] == "Hi"
    assert data["cues"][0]["source_text"] == "Hola"


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="Unknown output format"):
        render(Subtitles(cues=[]), "wav")


# --------------------------------------------------------------------------
# translator response validation
# --------------------------------------------------------------------------

from subtrans.translate.llm import _loose_json, _parse_response  # noqa: E402


def test_parse_response_accepts_well_formed_json():
    raw = '{"translations":[{"id":1,"text":"Hola"},{"id":2,"text":"Adiós"}]}'
    parsed, terms, problem = _parse_response(raw, [1, 2])
    assert parsed == {1: "Hola", 2: "Adiós"}
    assert problem == ""


def test_parse_response_rejects_missing_ids():
    raw = '{"translations":[{"id":1,"text":"Hola"}]}'
    parsed, _, problem = _parse_response(raw, [1, 2])
    assert parsed is None
    assert "missing" in problem


def test_parse_response_rejects_empty_translation():
    raw = '{"translations":[{"id":1,"text":"Hola"},{"id":2,"text":"   "}]}'
    parsed, _, problem = _parse_response(raw, [1, 2])
    assert parsed is None


def test_parse_response_rescues_positional_replies():
    raw = '{"translations":[{"text":"Hola"},{"text":"Adiós"}]}'
    parsed, _, _ = _parse_response(raw, [7, 8])
    assert parsed == {7: "Hola", 8: "Adiós"}


def test_parse_response_drops_extra_ids():
    raw = '{"translations":[{"id":1,"text":"A"},{"id":2,"text":"B"},{"id":9,"text":"X"}]}'
    parsed, _, _ = _parse_response(raw, [1, 2])
    assert parsed == {1: "A", 2: "B"}


def test_parse_response_collects_glossary_terms():
    raw = '{"translations":[{"id":1,"text":"Hola"}],"terms":{"Smith":"Smith"}}'
    _, terms, _ = _parse_response(raw, [1])
    assert terms == {"Smith": "Smith"}


def test_loose_json_handles_fences_and_truncation():
    assert _loose_json('```json\n{"a": 1}\n```') == {"a": 1}
    salvaged = _loose_json('{"translations":[{"id":1,"text":"Hola"}')
    assert salvaged is not None
    assert salvaged["translations"][0]["text"] == "Hola"
    assert _loose_json("not json at all") is None


# --------------------------------------------------------------------------
# media (real ffmpeg, real file)
# --------------------------------------------------------------------------

def _make_media(tmp_path: Path) -> Path:
    """A real 3-second WAV via ffmpeg's tone generator."""
    out = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3", "-ac", "1", "-ar", "44100", str(out)],
        check=True,
    )
    return out


def test_probe_reads_real_audio(tmp_path: Path):
    from subtrans.media import probe

    info = probe(_make_media(tmp_path))
    assert info.has_audio is True
    assert info.has_video is False
    assert info.duration == pytest.approx(3.0, abs=0.2)
    assert info.kind == "audio"


def test_extract_audio_produces_16k_mono(tmp_path: Path):
    from subtrans.media import extract_audio, probe

    wav = extract_audio(_make_media(tmp_path), dest=str(tmp_path / "out.wav"))
    info = probe(wav)
    assert info.sample_rate == 16000
    assert info.channels == 1


def test_probe_raises_on_garbage(tmp_path: Path):
    from subtrans.media import MediaError, probe

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is definitely not a video")
    with pytest.raises(MediaError):
        probe(bad)


def test_extract_audio_rejects_silent_video(tmp_path: Path):
    from subtrans.media import MediaError, extract_audio

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=black:s=64x64:r=5:d=1", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    with pytest.raises(MediaError, match="no audio stream"):
        extract_audio(silent, dest=str(tmp_path / "x.wav"))


# --------------------------------------------------------------------------
# pipeline (no model weights needed: subtitle in, subtitle/text out)
# --------------------------------------------------------------------------

def test_pipeline_converts_subtitle_file_without_any_backend(tmp_path: Path):
    from subtrans.pipeline import JobSpec, run

    src = tmp_path / "in.srt"
    src.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nSecond line here.\n",
        encoding="utf-8",
    )
    result = run(
        JobSpec(
            input_path=str(src), source_language="en",
            formats=["srt", "vtt", "txt", "json"],
            output_dir=str(tmp_path / "out"),
        )
    )
    assert result.kind == "subtitle"
    assert len(result.artifacts) == 4
    for artifact in result.artifacts:
        assert Path(artifact.path).exists()
        assert artifact.size > 0
    text = (tmp_path / "out" / "in.en.txt").read_text(encoding="utf-8")
    assert "Hello there." in text


def test_pipeline_reports_missing_file():
    from subtrans.pipeline import JobSpec, run
    from subtrans.types import MediaError

    with pytest.raises(MediaError, match="No such file"):
        run(JobSpec(input_path="/definitely/not/here.mp4"))


def test_every_format_renders_with_the_full_pipeline_kwarg_set():
    """`render` is called by the pipeline with a superset of kwargs; each
    writer must tolerate the ones it does not use."""
    from subtrans.formats.writers import EXTENSIONS

    subs = Subtitles(
        cues=[
            Cue(index=1, start=0.0, end=2.0, lines=["Line one"], speaker="Alex"),
            Cue(index=2, start=4.0, end=6.0, lines=["Line two", "wrapped"]),
        ],
        language="en",
        source_language="es",
        title="Sample",
    )
    pipeline_kwargs = {
        "timestamps": True,
        "speakers": True,
        "transcript": Transcript(language="en", duration=6.0, segments=[], model="m"),
        "metrics": {"cues": 2},
    }
    for fmt in EXTENSIONS:
        out = render(subs, fmt, **pipeline_kwargs)
        assert out and out.strip(), f"{fmt} produced nothing"
        assert "Line one" in out or "Line" in out, f"{fmt} lost the text"
