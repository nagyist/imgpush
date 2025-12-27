import contextlib
import os
import tempfile

import cv2
import settings

if settings.NUDE_FILTER_MAX_THRESHOLD:
    from nudenet import NudeClassifier

    nude_classifier = NudeClassifier()
else:
    nude_classifier = None


def get_video_duration(filepath: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(filepath)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            return 0.0
        return frame_count / fps
    finally:
        cap.release()


def check_video_duration(filepath: str) -> bool:
    """Check if video exceeds maximum duration.

    Returns True if video exceeds MAX_VIDEO_DURATION.
    """
    duration = get_video_duration(filepath)
    return duration > settings.MAX_VIDEO_DURATION


def extract_video_frames(filepath: str, interval: float, max_frames: int = 0) -> list[str]:
    """Extract frames from video at specified interval in seconds.

    Returns list of temporary file paths containing extracted frames.
    Caller is responsible for cleaning up these files.

    Args:
        filepath: Path to video file.
        interval: Interval in seconds between extracted frames.
        max_frames: Maximum number of frames to extract. 0 means no limit.
    """
    frame_paths: list[str] = []
    cap = cv2.VideoCapture(filepath)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return frame_paths

        frame_interval = int(fps * interval)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        while True:
            if max_frames > 0 and len(frame_paths) >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cv2.imwrite(temp_path, frame)
                frame_paths.append(temp_path)

            frame_count += 1
    finally:
        cap.release()

    return frame_paths


def check_video_nudity_filter(filepath: str) -> bool:
    """Check if video passes nudity filter by sampling frames.

    Returns True if ANY frame exceeds the nudity threshold.
    """
    if not settings.NUDE_FILTER_MAX_THRESHOLD or nude_classifier is None:
        return False

    interval = settings.NUDE_FILTER_VIDEO_INTERVAL
    max_frames = settings.NUDE_FILTER_MAX_FRAMES

    # Adjust interval to evenly distribute frames across video if max_frames is set
    if max_frames > 0:
        duration = get_video_duration(filepath)
        if duration > 0:
            min_interval_for_coverage = duration / max_frames
            interval = max(interval, min_interval_for_coverage)

    frame_paths = extract_video_frames(filepath, interval, max_frames)

    if not frame_paths:
        return False

    try:
        # Batch classify all frames at once for better performance
        results = nude_classifier.classify(frame_paths)
        for frame_path in frame_paths:
            unsafe_val = results.get(frame_path, {}).get("unsafe", 0)
            if unsafe_val >= settings.NUDE_FILTER_MAX_THRESHOLD:
                return True
        return False
    finally:
        for frame_path in frame_paths:
            with contextlib.suppress(OSError):
                os.remove(frame_path)
