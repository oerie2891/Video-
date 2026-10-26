"""Command-line interface.

    subtrans talk.mp4                        # transcribe -> talk.en.srt
    subtrans talk.mp4 --to es fr ja          # + translated subtitles
    subtrans talk.mp4 -f srt vtt txt md      # several outputs at once
    subtrans existing.srt --to de            # translate subtitles, keep timings
    subtrans report.docx --to ja -f docx     # translate a document
    subtrans serve                           # web UI on :8000
    subtrans languages --search port         # what can this do?
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .asr import available_backends as asr_backends
from .asr.faster_whisper_backend import MODEL_SIZES
from .formats import EXTENSIONS
from .languages import LANGUAGES, display_name, normalize, quality_tier
from .pipeline import JobResult, JobSpec, run
from .translate import available_backends as mt_backends
from .types import BackendError, MediaError

# --------------------------------------------------------------------------
# Progress rendering
# --------------------------------------------------------------------------

class _Bar:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.last = 0.0
        self.started = time.monotonic()

    def __call__(self, fraction: float, message: str) -> None:
        if not self.enabled:
            # Non-TTY: one line per meaningful step, no control characters.
            if fraction - self.last >= 0.1 or fraction >= 1.0:
                self.last = fraction
                print(f"[{fraction * 100:5.1f}%] {message}", file=sys.stderr, flush=True)
            return
        width = max(20, min(44, (shutil.get_terminal_size((80, 20)).columns) - 42))
        filled = int(width * min(1.0, max(0.0, fraction)))
        bar = "━" * filled + "╌" * (width - filled)
        elapsed = time.monotonic() - self.started
        eta = ""
        if 0.02 < fraction < 1.0:
            remaining = elapsed * (1 - fraction) / fraction
            eta = f" eta {_short(remaining)}"
        line = f"\r  {bar} {fraction * 100:5.1f}%{eta}  {message}"
        cols = shutil.get_terminal_size((80, 20)).columns
        sys.stderr.write(line[: cols - 1].ljust(cols - 1))
        sys.stderr.flush()
        if fraction >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _short(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtrans",
        description=(
            "Generate subtitles and transcripts from any video or audio file, "
            "and translate video, audio, subtitle or text files into any language."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"subtrans {__version__}")
    sub = parser.add_subparsers(dest="command")

    job = sub.add_parser("run", help="Transcribe and/or translate a file (default)")
    _add_job_arguments(job)

    serve = sub.add_parser("serve", help="Start the web UI and HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1, help="Concurrent jobs")
    serve.add_argument(
        "--data-dir", default=None, help="Where uploads and results are kept"
    )

    langs = sub.add_parser("languages", help="List supported languages")
    langs.add_argument("--search", default=None, help="Filter by name or code")

    sub.add_parser("backends", help="Show which ASR/translation backends are usable")
    return parser


def _add_job_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Video, audio, subtitle (.srt/.vtt/.ass) or text file")
    p.add_argument(
        "--to", "-t", nargs="+", default=[], metavar="LANG",
        help="Target language(s): code or English name. Repeatable.",
    )
    p.add_argument(
        "--source", "-s", default=None, metavar="LANG",
        help="Source language. Omit to auto-detect (recommended only for clear audio).",
    )
    p.add_argument(
        "--format", "-f", nargs="+", default=["srt"], metavar="FMT",
        choices=sorted(set(EXTENSIONS) | {"docx"}),
        help=f"Output format(s): {', '.join(sorted(set(EXTENSIONS) | {'docx'}))}",
    )
    p.add_argument("--out", "-o", default=None, help="Output directory")
    p.add_argument("--name", default=None, help="Output filename stem")

    asr = p.add_argument_group("transcription")
    asr.add_argument(
        "--model", "-m", default=None, metavar="SIZE",
        help=f"Whisper model: {', '.join(MODEL_SIZES[:12])}… (default: small)",
    )
    asr.add_argument("--asr-backend", default="auto", help="auto | local | openai")
    asr.add_argument("--device", default=None, help="cpu | cuda")
    asr.add_argument("--compute-type", default=None, help="int8 | float16 | float32")
    asr.add_argument("--beam-size", type=int, default=5)
    asr.add_argument(
        "--no-vad", action="store_true",
        help="Disable voice-activity filtering (keep if the audio has long silences)",
    )
    asr.add_argument(
        "--no-normalize", action="store_true",
        help="Skip the speech levelling filter applied before transcription",
    )
    asr.add_argument(
        "--prompt", default=None, metavar="TEXT",
        help="Bias the model toward names/jargon, e.g. 'Kubernetes, Grafana, Loki'",
    )

    mt = p.add_argument_group("translation")
    mt.add_argument("--mt-backend", default="auto", help="auto | llm | anthropic | openai | nllb")
    mt.add_argument("--mt-model", default=None, help="Translation model id")
    mt.add_argument(
        "--window", type=int, default=24,
        help="Cues translated per request (more context, fewer calls)",
    )
    mt.add_argument("--context", type=int, default=6, help="Preceding cues sent as context")
    mt.add_argument("--formality", choices=["formal", "informal"], default=None)
    mt.add_argument(
        "--notes", default=None,
        help="Domain guidance for the translator, e.g. 'medical lecture, keep Latin terms'",
    )
    mt.add_argument(
        "--glossary", default=None, metavar="FILE",
        help="JSON or TSV file of source->target terms to enforce",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--no-source", action="store_true",
        help="Write only translations, not the original-language files",
    )
    out.add_argument("--no-timestamps", action="store_true", help="Plain text without timecodes")
    out.add_argument("--no-speakers", action="store_true", help="Omit speaker labels")
    out.add_argument("--burn-in", action="store_true", help="Also render an MP4 with burned subtitles")
    out.add_argument("--quiet", "-q", action="store_true")
    out.add_argument("--json", action="store_true", help="Print a machine-readable summary")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def _load_glossary(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"subtrans: glossary file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        import json

        data = json.loads(text)
        if not isinstance(data, dict):
            raise SystemExit("subtrans: glossary JSON must be an object of term -> term")
        return {str(k): str(v) for k, v in data.items()}
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(",", 1)
        if len(parts) >= 2:
            out[parts[0].strip()] = parts[1].strip()
    return out


def cmd_run(args: argparse.Namespace) -> int:
    try:
        targets = [normalize(t) or t for t in args.to]
        source = normalize(args.source)
    except ValueError as exc:
        print(f"subtrans: {exc}", file=sys.stderr)
        return 2

    spec = JobSpec(
        input_path=args.input,
        targets=targets,
        source_language=source,
        formats=args.format,
        output_dir=args.out,
        output_stem=args.name,
        asr_backend=args.asr_backend,
        asr_model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        vad=not args.no_vad,
        normalize_audio=not args.no_normalize,
        initial_prompt=args.prompt,
        beam_size=args.beam_size,
        mt_backend=args.mt_backend,
        mt_model=args.mt_model,
        window=args.window,
        context=args.context,
        formality=args.formality,
        glossary=_load_glossary(args.glossary),
        notes=args.notes,
        include_source=not args.no_source,
        timestamps_in_text=not args.no_timestamps,
        speakers_in_output=not args.no_speakers,
        burn_in=args.burn_in,
    )

    bar = _Bar(enabled=sys.stderr.isatty() and not args.quiet)
    progress = (lambda f, m: None) if args.quiet else bar
    try:
        result = run(spec, progress)
    except (MediaError, BackendError) as exc:
        print(f"\nsubtrans: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nsubtrans: interrupted", file=sys.stderr)
        return 130

    if args.json:
        import json

        print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    elif not args.quiet:
        _print_summary(result)
    return 0


def _summary(result: JobResult) -> dict:
    return {
        "kind": result.kind,
        "source_language": result.source_language,
        "elapsed_seconds": round(result.elapsed, 2),
        "languages": sorted(set(result.subtitles) | set(result.documents)),
        "artifacts": [
            {"language": a.language, "format": a.format, "path": a.path, "bytes": a.size}
            for a in result.artifacts
        ],
        "metrics": result.metrics,
        "warnings": result.warnings,
        "glossary": result.glossary,
    }


def _print_summary(result: JobResult) -> None:
    if result.source_language:
        tier = quality_tier(result.source_language)
        print(
            f"\n  Source: {display_name(result.source_language)} "
            f"({result.source_language}, accuracy tier: {tier})"
        )
    if result.transcript:
        t = result.transcript
        print(f"  Model:  {t.model} via {t.backend}")

    for language, metrics in result.metrics.items():
        if not metrics.get("cues"):
            continue
        flags = []
        if metrics.get("over_cps"):
            flags.append(f"{metrics['over_cps']} fast")
        if metrics.get("over_length"):
            flags.append(f"{metrics['over_length']} over line length")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  {language:>5}: {metrics['cues']} cues, "
            f"{metrics['mean_cps']} cps avg (peak {metrics['max_cps']}){suffix}"
        )

    if result.artifacts:
        print("\n  Written:")
        for a in result.artifacts:
            print(f"    {a.path}  ({_bytes(a.size)})")
    if result.warnings:
        print("\n  Notes:")
        for w in result.warnings:
            print(f"    - {w}")
    print(f"\n  Finished in {_short(result.elapsed)}.\n")


def _bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def cmd_languages(args: argparse.Namespace) -> int:
    query = (args.search or "").lower()
    rows = [
        (code, name, quality_tier(code))
        for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])
        if not query or query in name.lower() or query in code
    ]
    if not rows:
        print(f"No language matches {args.search!r}")
        return 1
    print(f"{'code':<6}{'language':<24}transcription accuracy")
    print("-" * 52)
    for code, name, tier in rows:
        print(f"{code:<6}{name:<24}{tier}")
    print(
        f"\n{len(rows)} languages. Transcription accuracy is Whisper's; translation "
        f"targets are not limited to this list when using the LLM backend."
    )
    return 0


def cmd_backends(_: argparse.Namespace) -> int:
    print("Transcription backends:")
    for info in asr_backends():
        mark = "ready" if info["available"] else "unavailable"
        print(f"  {info['name']:<16} {mark:<12} {info['reason']}")
    print("\nTranslation backends:")
    for info in mt_backends():
        mark = "ready" if info["available"] else "unavailable"
        model = f"({info.get('model')})" if info.get("model") else ""
        print(f"  {info['name']:<16} {mark:<12} {info['reason']} {model}")
    print(
        "\nFor the best translation quality set ANTHROPIC_API_KEY (or OPENAI_API_KEY).\n"
        "Without a key, transcription still works fully offline, and Whisper can\n"
        "translate speech into English only."
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "subtrans: the web UI needs fastapi and uvicorn\n"
            "  pip install 'subtrans[server]'",
            file=sys.stderr,
        )
        return 1
    if args.data_dir:
        os.environ["SUBTRANS_DATA_DIR"] = args.data_dir
    os.environ["SUBTRANS_WORKERS"] = str(max(1, args.workers))

    from .server import create_app

    print(f"subtrans UI → http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Allow `subtrans FILE ...` without typing the `run` subcommand.
    commands = {"run", "serve", "languages", "backends"}
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "languages":
        return cmd_languages(args)
    if args.command == "backends":
        return cmd_backends(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
