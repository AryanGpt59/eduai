"""Conservative, rule-based facilitation prompts for session review."""

from __future__ import annotations


def teaching_prompts(
    *,
    confirmed_tracks: int,
    peak_faces: int,
    mean_motion: float,
    audio_activity: float | None,
) -> list[str]:
    """Suggest teacher check-ins from aggregate signals, with no student judgments."""
    prompts: list[str] = []
    if peak_faces == 0:
        prompts.append("No faces were detected in the sampled frames. Check camera framing or use a different clip.")
    elif confirmed_tracks == 0:
        prompts.append("No anonymous track reached the selected presence threshold. Review the clip and threshold settings.")
    if peak_faces and mean_motion < 0.08:
        prompts.append("The sampled video showed little tracked movement. If it fits the lesson, consider a brief question or activity to check in with the class.")
    if audio_activity is not None and audio_activity > 0.55:
        prompts.append("The measured audio level was elevated in some windows. Review the timeline in context; loudness alone does not identify participation.")
    if not prompts:
        prompts.append("Review the attendance and signal timelines alongside your own classroom observations before deciding whether to change the lesson.")
    prompts.append("These are rule-based review prompts. Movement and sound are not measures of attention, understanding, or learning.")
    return prompts
