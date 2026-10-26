# subtrans

Generate subtitles and transcripts from any video or audio file, and translate
video, audio, subtitle or text files into any language.

Works offline by default. A web UI and a CLI share the same core.

```bash
subtrans talk.mp4                      # -> talk.en.srt
subtrans talk.mp4 --to es ja -f srt md # -> + talk.es.srt, talk.ja.srt, …
subtrans existing.srt --to de          # translate, keep the original timings
subtrans report.docx --to fr -f docx   # translate a document
subtrans serve                         # web UI on http://127.0.0.1:8000
```

## Why this isn't just a Whisper wrapper

Three things decide whether auto-subtitles are usable, and they are the three
things most tools skip:

**1. Cue segmentation.** Whisper emits segments of whatever length its decoder
happened to produce — sometimes one word, sometimes thirty seconds of run-on
text. Writing those straight to SRT is the most common mistake in this
category. `subtrans` re-segments from word-level timestamps: words become
sentences, sentences are split into cues by dynamic programming that balances
cue length while preferring breaks at punctuation and natural pauses, and each
cue is wrapped into balanced lines that never orphan a function word
("in the / kitchen").

**2. Context-aware translation.** A subtitle cue is usually half a sentence.
Translating cues one at a time — which is what a per-cue call to any MT API
does — loses the subject, the tense and the register, and a sentence spread
across three cues comes back as three unrelated fragments. `subtrans` sends a
rolling window of cues *plus the preceding cues and their accepted
translations*, requires the same number of units back in the same order, and
accumulates a glossary so names and terminology stay identical from the first
cue to the last.

**3. Honest reporting.** Reading speed (characters per second), over-long
lines, low language-detection confidence and the accuracy tier of the detected
language are all reported rather than hidden. Whisper nominally supports ~99
languages; its error rate across them varies by an order of magnitude, and you
should know which end you are on before publishing.

Also handled: silence hallucinations (Whisper's "Thanks for watching!" loops are
detected and dropped), per-language line-length conventions (16 characters for
CJK, 42 for Latin scripts), kinsoku line breaking for scripts without spaces,
orphan-cue merging, and timing hygiene (no overlaps, minimum durations,
inter-cue gaps).

## Install

Requires Python 3.10+ and **ffmpeg** on your PATH.

```bash
git clone https://github.com/oerie2891/video-.git && cd video-
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e '.[openai]'     # cloud transcription / OpenAI translation
pip install -e '.[local-mt]'   # offline translation (NLLB-200, ~2.5 GB + torch)
pip install -e '.[dev]'        # pytest, ruff
```

### Transcription works with no keys

The default backend is `faster-whisper`, which runs locally. The first run
downloads the model from Hugging Face (`small` is ~500 MB, `large-v3` ~3 GB),
after which it needs no network at all.

On CPU, expect roughly 1–3× realtime for `large-v3` with int8 — a one-hour
video takes 20–60 minutes on four cores. `large-v3-turbo` is close in quality
and several times faster; `distil-large-v3` is faster still for English.

### Translation needs a key for good results

| Backend | Quality for subtitles | Requirement |
|---|---|---|
| `llm` (Anthropic / OpenAI) | best — this is the one that handles cue context | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| `nllb` (local) | fair — no cross-sentence context, approximate re-cueing | `pip install '.[local-mt]'` |
| Whisper native | English only, speech→English in one pass | nothing |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
subtrans backends          # shows what is usable right now, and why not
```

Without any translation backend, transcription still works fully, and Whisper
can translate speech into English (only English) by itself.

## Usage

### CLI

```bash
# Transcribe, pick the model explicitly, force the source language
subtrans interview.m4a --model large-v3 --source en -f srt vtt txt

# Translate into several languages in one pass; glossary keeps names consistent
subtrans keynote.mp4 --to es pt-BR ja --glossary terms.tsv -f srt

# Bias the model toward jargon it would otherwise mangle
subtrans standup.mp4 --prompt "Kubernetes, Grafana, Loki, Ana Sofía"

# Give the translator domain context
subtrans lecture.mp4 --to de --notes "Medical lecture; keep Latin anatomy terms"

# Translate an existing subtitle file, preserving every timing
subtrans episode.vtt --to fr -f vtt srt

# Burn subtitles into a video
subtrans clip.mp4 --to es -f srt --burn-in

# Machine-readable output for scripting
subtrans talk.mp4 --json -q
```

Useful subcommands:

```bash
subtrans languages --search port   # supported languages + accuracy tier
subtrans backends                  # what works right now
subtrans serve --port 8080 --workers 2
```

### Web UI

```bash
subtrans serve
```

Drag in a file, pick target languages and formats, watch live progress, preview
the cues with reading-speed flags, download the results. Uploads are deleted
once a job finishes; outputs are kept for the session.

### Python

```python
from subtrans import JobSpec, run, transcribe, translate_subtitles

# One-stop
result = run(JobSpec(
    input_path="interview.mp4",
    targets=["es", "ja"],
    formats=["srt", "vtt", "md"],
))
for artifact in result.artifacts:
    print(artifact.language, artifact.path)
for warning in result.warnings:
    print("note:", warning)

# Or step by step
transcript, subs = transcribe("talk.mp3")
spanish = translate_subtitles(subs, "es", source=transcript.language)
print(spanish.cues[0].lines)
```

## Formats

**Input** — video (`.mp4 .mkv .mov .avi .webm .flv .wmv .ts .mpg .3gp` …),
audio (`.mp3 .wav .m4a .flac .ogg .opus .aac .wma` …), subtitles
(`.srt .vtt .ass .ssa .sbv`), documents (`.txt .md .docx .json`). Anything
ffmpeg can decode works, extension or not.

**Output** — `srt`, `vtt`, `ass`, `txt` (reflowed into paragraphs, not cue
soup), `md`, `json` (cues, timings, reading speed, source text), `tsv`, `lrc`,
`docx`.

## Options worth knowing

| Flag | Why |
|---|---|
| `--source LANG` | Auto-detection is unreliable on noisy or short audio. Force it. |
| `--model large-v3-turbo` | Best quality-per-second on CPU. |
| `--prompt "names, jargon"` | Biases the decoder; markedly improves proper nouns. |
| `--notes "..."` | Domain context for the translator. |
| `--glossary FILE` | JSON or TSV of enforced term translations. |
| `--window N` | Cues per translation request. Larger = more context, fewer calls. |
| `--no-vad` | Keep if the audio has long musical or quiet passages you still want transcribed. |
| `--no-normalize` | Skip the speech-levelling filter for already-mastered audio. |

## Environment variables

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Enables LLM translation |
| `SUBTRANS_WHISPER_MODEL` | Default model size (`small`) |
| `SUBTRANS_LLM_MODEL` | Default translation model |
| `SUBTRANS_MODEL_DIR` | Where Whisper weights are cached |
| `SUBTRANS_ASR_BACKEND` / `SUBTRANS_MT_BACKEND` | Pin a backend |
| `SUBTRANS_DATA_DIR` | Where the server keeps jobs |
| `SUBTRANS_MAX_UPLOAD_MB` | Upload limit (default 4096) |

## Architecture

```
subtrans/
  media.py          ffmpeg: probe, extract to 16 kHz mono WAV, burn-in
  asr/              transcription backends (faster-whisper, OpenAI-compatible)
  cues.py           words -> sentences -> balanced cues; hallucination filter;
                    timing hygiene; reading-speed metrics
  linebreak.py      line wrapping (variance + linguistic penalties; kinsoku)
  translate/        translation backends (context-windowed LLM, local NLLB)
  formats/          readers and writers for every format
  languages.py      language table, per-language cue style, accuracy tiers
  pipeline.py       orchestration shared by CLI and server
  jobs.py           in-process job queue with progress and cancellation
  server.py         FastAPI app
  web/index.html    single-file UI, no build step
```

Backends sit behind narrow interfaces (`asr/base.py`, `translate/base.py`): a
backend sees audio or text units and nothing about cues or formats, which is
what lets a local model and a cloud API swap without touching anything else.

## Limitations

Stated plainly rather than discovered later:

- **Speaker diarization is not implemented.** The data model carries speaker
  labels through every layer, and the writers emit them, but nothing assigns
  them. Wiring in `pyannote.audio` or a cloud ASR that returns speakers is the
  natural next step.
- **Accuracy varies enormously by language.** `subtrans languages` shows the
  tier. Treat `limited` as draft-only.
- **The local NLLB path redistributes a translated sentence across its cues
  proportionally**, which is approximate and degrades for pairs with very
  different word order (English→Japanese). The LLM path does not have this
  problem.
- **The job queue is in-process and not durable.** Restarting the server loses
  queued work. Swap `jobs.py` for Celery or RQ if you need persistence.
- **Translation cost scales with cue count**, not file size — roughly one
  request per `--window` cues.

## Development

```bash
pip install -e '.[dev]'
pytest -q          # 56 tests
ruff check subtrans tests
```

The test suite deliberately concentrates on cue segmentation, line breaking,
timing hygiene, format round-trips and translator-response validation — the
logic that decides output quality — plus the HTTP API end to end. Those tests
need neither model weights nor an API key.

## License

MIT
