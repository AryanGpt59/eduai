"""Streamlit interface for the independent EduAI demo."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from audio import analyze_wav_bytes
from history import clear_summaries, list_summaries, save_summary
from insights import teaching_prompts
from video import analyze_video


st.set_page_config(page_title="EduAI", page_icon="🎓", layout="wide")
st.title("EduAI")
st.caption("Classroom presence and activity review")
st.info(
    "This demo reports anonymous face-location tracks, movement proxies, and optional audio levels. "
    "It does not identify students or measure attention, understanding, or learning."
)
st.warning(
    "Upload notice: selecting video or audio sends it to the machine running this app. In the hosted demo, "
    "that is Streamlit Community Cloud. Files may contain identifiable students; use only media you are "
    "authorized to share, and obtain required consent. Video is copied temporarily for analysis and deleted "
    "afterward; saved summaries contain aggregate metrics only."
)

page = st.sidebar.radio("View", ["Analyze a session", "Saved summaries"])

if page == "Saved summaries":
    st.subheader("Saved summaries")
    st.caption(
        "Only aggregate counts and settings are stored. Video, audio, images, and track IDs are not saved here. "
        "In the hosted demo, this database is on the Streamlit server and is not a durable archive."
    )
    saved = list_summaries()
    if saved:
        st.dataframe(pd.DataFrame(saved), use_container_width=True, hide_index=True)
    else:
        st.write("No saved summaries yet.")
    if st.checkbox("I understand this permanently clears saved summaries on this device"):
        if st.button("Clear local summaries", type="secondary"):
            clear_summaries()
            st.success("Saved summaries cleared.")
            st.rerun()
else:
    st.subheader("Analyze a video")
    cloud_acknowledged = st.checkbox(
        "I understand that selecting media sends it to the server running this app for analysis.",
        key="cloud_upload_acknowledged",
    )
    if cloud_acknowledged:
        video_file = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
        audio_file = st.file_uploader("Optional separate audio file (WAV/FLAC/OGG/MP3)", type=["wav", "flac", "ogg", "mp3"])
    else:
        video_file = None
        audio_file = None
        st.caption("Confirm the data notice to reveal the media upload controls.")
    col_a, col_b = st.columns(2)
    with col_a:
        confirmation_seconds = st.slider("Observed-time attendance threshold (seconds)", 1, 60, 3)
    with col_b:
        max_samples = st.slider("Maximum sampled video frames", 100, 1800, 900, step=100)
    st.caption("Video is sampled locally at about 2 frames per second. Attendance labels are temporary anonymous tracks.")

    if st.button("Analyze session", type="primary", disabled=video_file is None):
        st.session_state.pop("last_analysis", None)
        suffix = Path(video_file.name).suffix or ".mp4"
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(prefix="eduai-", suffix=suffix, delete=False) as temp_video:
                temp_video.write(video_file.getbuffer())
                temp_name = temp_video.name
            with st.spinner("Analyzing sampled frames on this device…"):
                timeline, attendance, metadata = analyze_video(
                    temp_name,
                    confirmation_seconds=confirmation_seconds,
                    max_samples=max_samples,
                )
            audio_rows = analyze_wav_bytes(audio_file.getvalue()) if audio_file is not None else []
            timeline_df = pd.DataFrame(timeline)
            attendance_df = pd.DataFrame(attendance)
            audio_df = pd.DataFrame(audio_rows)
            summary = {
                "duration_seconds": metadata["duration_seconds"],
                "sampled_frames": int(metadata["sampled_frames"]),
                "peak_detected_faces": int(timeline_df["detected_faces"].max()),
                "confirmed_tracks": int((attendance_df["attendance"] == "Confirmed").sum()) if not attendance_df.empty else 0,
                "audio_supplied": audio_file is not None,
                "confirmation_seconds": confirmation_seconds,
            }
            st.session_state["last_analysis"] = {
                "timeline": timeline_df,
                "attendance": attendance_df,
                "audio": audio_df,
                "metadata": metadata,
                "summary": summary,
            }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)

    result = st.session_state.get("last_analysis")
    if result:
        timeline_df = result["timeline"]
        attendance_df = result["attendance"]
        audio_df = result["audio"]
        summary = result["summary"]
        st.divider()
        st.subheader("Session overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Video duration", f"{result['metadata']['duration_seconds']:.1f} s")
        m2.metric("Sampled frames", f"{int(result['metadata']['sampled_frames'])}")
        m3.metric("Peak face detections", f"{summary['peak_detected_faces']}")
        m4.metric("Tracks above threshold", f"{summary['confirmed_tracks']}")

        st.subheader("Observed signal timeline")
        st.caption("These are technical measurements, not a student engagement or learning score.")
        visual_chart = timeline_df.set_index("seconds")[["detected_faces", "motion_activity_proxy"]]
        st.line_chart(visual_chart)
        if not audio_df.empty:
            st.line_chart(audio_df.set_index("seconds")[["activity_level"]])
        else:
            st.caption("No audio signal was supplied; audio is excluded from this session.")

        st.subheader("Anonymous attendance tracks")
        if attendance_df.empty:
            st.write("No face tracks were detected in the sampled frames.")
        else:
            st.dataframe(attendance_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download attendance CSV",
            attendance_df.to_csv(index=False).encode("utf-8"),
            file_name="eduai-attendance.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download video timeline CSV",
            timeline_df.to_csv(index=False).encode("utf-8"),
            file_name="eduai-video-timeline.csv",
            mime="text/csv",
        )
        if not audio_df.empty:
            st.download_button(
                "Download audio activity CSV",
                audio_df.to_csv(index=False).encode("utf-8"),
                file_name="eduai-audio-activity.csv",
                mime="text/csv",
            )

        peak_faces = summary["peak_detected_faces"]
        mean_motion = float(timeline_df["motion_activity_proxy"].mean())
        mean_audio = float(audio_df["activity_level"].mean()) if not audio_df.empty else None
        st.subheader("Rule-based teaching prompts")
        for prompt in teaching_prompts(
            confirmed_tracks=summary["confirmed_tracks"],
            peak_faces=peak_faces,
            mean_motion=mean_motion,
            audio_activity=mean_audio,
        ):
            st.write(f"• {prompt}")

        if st.button("Save aggregate summary to this app"):
            save_summary(summary)
            st.success("Aggregate summary saved to this app's local database.")

st.divider()
st.caption("Keep teacher judgment central. Review consent, retention, and classroom policy before any real-world use.")
