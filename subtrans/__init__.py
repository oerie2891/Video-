"""subtrans — subtitles, transcripts and translation for any media or text file.

Quick start:

    from subtrans import JobSpec, run

    result = run(JobSpec(
        input_path="interview.mp4",
        targets=["es", "ja"],
        formats=["srt", "vtt", "md"],
    ))
    for artifact in result.artifacts:
        print(artifact.path)

Lower-level pieces are importable too:

    from subtrans import transcribe, translate_subtitles
    transcript, subs = transcribe("talk.mp3")
    spanish = translate_subtitles(subs, "es")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .languages import LANGUAGES, display_name, normalize, quality_tier
from .pipeline import (
    Artifact,
    JobResult,
    JobSpec,
    run,
    transcribe,
    translate_blocks,
    translate_subtitles,
)
from .types import BackendError, Cue, MediaError, Segment, Subtitles, Transcript, Word

__all__ = [
    "__version__",
    "Artifact",
    "BackendError",
    "Cue",
    "JobResult",
    "JobSpec",
    "LANGUAGES",
    "MediaError",
    "Segment",
    "Subtitles",
    "Transcript",
    "Word",
    "display_name",
    "normalize",
    "quality_tier",
    "run",
    "transcribe",
    "translate_blocks",
    "translate_subtitles",
]
