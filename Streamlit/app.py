"""People Counting Assistant (Distracted Pedestrian Detection).

Streamlit dashboard for campus walkway monitoring. A fine-tuned YOLO26n model
(``models/best.pt``) detects four classes:

    0  person_left        pedestrian walking left
    1  person_right       pedestrian walking right
    2  phone_using_left   distracted pedestrian (phone) walking left
    3  phone_using_right  distracted pedestrian (phone) walking right

The app reports directional counts, a distraction rate with a safety badge,
an annotated image, a per-class breakdown and a downloadable JSON payload.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_TITLE: str = "People Counting Assistant"
APP_SUBTITLE: str = (
    "Distracted Pedestrian Detection for campus walkways. Counts pedestrians by "
    "walking direction (left / right) and flags those using a phone while walking."
)

MODEL_PATH: Path = Path(__file__).resolve().parent / "models" / "best.pt"
MODEL_VERSION: str = "1.0.0"
MODEL_ARCHITECTURE: str = "YOLO26n"

# Order must match the training data.yaml (IDs 0..3).
CLASS_NAMES: tuple[str, ...] = (
    "person_left",
    "person_right",
    "phone_using_left",
    "phone_using_right",
)
LEFT_CLASSES: frozenset[str] = frozenset({"person_left", "phone_using_left"})
RIGHT_CLASSES: frozenset[str] = frozenset({"person_right", "phone_using_right"})
DISTRACTED_CLASSES: frozenset[str] = frozenset({"phone_using_left", "phone_using_right"})

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "person_left": "Person · Left",
    "person_right": "Person · Right",
    "phone_using_left": "Phone Using · Left",
    "phone_using_right": "Phone Using · Right",
}

# RGB colours (the image array is RGB throughout the app).
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "person_left": (52, 120, 246),       # blue
    "person_right": (16, 172, 132),      # green
    "phone_using_left": (245, 130, 32),  # orange
    "phone_using_right": (220, 38, 38),  # red
}
FALLBACK_COLOR_RGB: tuple[int, int, int] = (128, 128, 128)

DEFAULT_CONFIDENCE: float = 0.40
DEFAULT_IOU: float = 0.45

ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png")
ALLOWED_PIL_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS: int = 40_000_000        # guards against decompression bombs

SAFE_RATE_LIMIT: float = 15.0     # below this: green
CAUTION_RATE_LIMIT: float = 30.0  # up to this: yellow, above: red

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

class Detection(TypedDict):
    """One detected object, in the JSON export format."""

    class_name: str
    confidence: float
    box: list[int]


@dataclass(frozen=True)
class SidebarSettings:
    """User-selected options from the sidebar."""

    confidence: float
    iou: float
    visible_classes: frozenset[str]
    show_labels: bool
    show_confidence: bool


@dataclass(frozen=True)
class InferenceResult:
    """Detections and timing for one image."""

    detections: list[Detection]
    inference_seconds: float


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable JPG/PNG image."""


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

CUSTOM_CSS: str = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .app-header h1 { margin-bottom: 0.25rem; font-weight: 700; }
    .app-header p  { color: #6b7280; font-size: 1.02rem; margin-top: 0; }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 5px solid var(--accent, #3478f6);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        height: 100%;
    }
    .kpi-label {
        color: #6b7280; font-size: 0.80rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .kpi-value { color: #111827; font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .kpi-sub   { color: #9ca3af; font-size: 0.80rem; }

    .badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; margin-left: 0.4rem;
        vertical-align: middle;
    }
    .badge-green  { background: #dcfce7; color: #166534; }
    .badge-yellow { background: #fef9c3; color: #854d0e; }
    .badge-red    { background: #fee2e2; color: #991b1b; }

    .model-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #f8fafc; border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.5rem;
    }
    .model-card .title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; }
    .model-card .row   { display: flex; justify-content: space-between; font-size: 0.84rem; }
    .model-card .row span:first-child { color: #cbd5e1; }

    .timing-pill {
        display: inline-block; background: #eef2ff; color: #3730a3;
        border-radius: 999px; padding: 0.25rem 0.8rem; font-size: 0.85rem; font-weight: 600;
    }

    .swatch {
        display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 3px;
        margin-right: 0.4rem; vertical-align: middle;
    }
</style>
"""


def inject_css() -> None:
    """Inject the app's custom CSS."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a ``#rrggbb`` string."""
    return "#{:02x}{:02x}{:02x}".format(*color)


# --------------------------------------------------------------------------- #
# Model loading and inference
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading YOLO26n model…")
def load_model(model_path: str) -> YOLO:
    """Load the YOLO model once per process and keep it in memory.

    Args:
        model_path: Path to the ``.pt`` weights file.

    Returns:
        The loaded Ultralytics YOLO model.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found at '{path}'.")
    return YOLO(str(path))


def model_class_names(model: YOLO) -> list[str]:
    """Return the model's class names ordered by class ID."""
    names = model.names
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


@st.cache_data(show_spinner=False, max_entries=32)
def run_inference(
    image_digest: str,
    _image: Image.Image,
    _model: YOLO,
    confidence: float,
    iou: float,
    model_version: str,
) -> tuple[list[Detection], float]:
    """Run YOLO inference on one image.

    Results are cached on ``image_digest``, ``confidence``, ``iou`` and
    ``model_version`` so that sidebar display toggles do not re-run the model.
    Arguments prefixed with ``_`` are excluded from the cache key. Plain
    built-in types are returned because ``st.cache_data`` pickles the result.

    Args:
        image_digest: SHA-256 of the uploaded bytes (cache key).
        _image: RGB PIL image.
        _model: Loaded YOLO model.
        confidence: Minimum confidence threshold.
        iou: NMS IoU threshold.
        model_version: Model version string (cache key).

    Returns:
        Tuple of (detections, inference time in seconds).
    """
    del image_digest, model_version  # only used as cache keys

    start = time.perf_counter()
    # A PIL image is passed (not a NumPy array) so Ultralytics treats it as RGB.
    results = _model.predict(source=_image, conf=confidence, iou=iou, verbose=False)
    elapsed = time.perf_counter() - start

    detections: list[Detection] = []
    if not results:
        return detections, elapsed

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections, elapsed

    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
        class_name = str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(class_id)
        detections.append(
            Detection(
                class_name=class_name,
                confidence=round(float(conf), 4),
                box=[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            )
        )

    detections.sort(key=lambda det: det["confidence"], reverse=True)
    return detections, elapsed


# --------------------------------------------------------------------------- #
# Image upload and validation
# --------------------------------------------------------------------------- #

def load_uploaded_image(file_name: str, raw: bytes) -> Image.Image:
    """Validate uploaded bytes and decode them into an RGB PIL image.

    Checks the file extension, size, real image format (from file content,
    not the extension), and pixel count. EXIF orientation is applied.

    Args:
        file_name: Original file name from the uploader.
        raw: File content.

    Returns:
        An RGB PIL image with EXIF rotation applied.

    Raises:
        ImageValidationError: If the file is not a valid JPG/PNG image.
    """
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '.{extension}'. Please upload a JPG, JPEG or PNG image."
        )
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"File is {len(raw) / 1_048_576:.1f} MB. The maximum size is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB."
        )

    try:
        # verify() checks file integrity without decoding all pixel data.
        with Image.open(io.BytesIO(raw)) as probe:
            detected_format = probe.format
            probe.verify()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("The image resolution is too large to process.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "The file could not be read as an image. It may be corrupted or not a real JPG/PNG."
        ) from exc

    if detected_format not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(
            f"The file content is '{detected_format}', not JPG or PNG. "
            "Please upload a genuine JPG, JPEG or PNG image."
        )

    try:
        # verify() leaves the image unusable, so the bytes are opened again.
        with Image.open(io.BytesIO(raw)) as decoded:
            image = ImageOps.exif_transpose(decoded).convert("RGB")
            image.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("The image resolution is too large to process.") from exc
    except (OSError, ValueError) as exc:
        raise ImageValidationError(f"The image could not be decoded: {exc}") from exc

    width, height = image.size
    if width < 1 or height < 1:
        raise ImageValidationError("The image has no pixels.")
    return image


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #

def compute_counts(detections: list[Detection]) -> dict[str, int]:
    """Count detections for each of the four known classes."""
    counts = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        if det["class_name"] in counts:
            counts[det["class_name"]] += 1
    return counts


def compute_summary(counts: dict[str, int]) -> dict[str, Any]:
    """Build totals, direction counts and the distraction rate.

    Args:
        counts: Per-class counts from :func:`compute_counts`.

    Returns:
        Dict with ``total_pedestrians``, ``total_distracted``,
        ``distraction_rate_pct``, ``walking_left`` and ``walking_right``.
    """
    total = sum(counts.values())
    distracted = sum(counts[name] for name in DISTRACTED_CLASSES)
    rate = round(distracted / total * 100, 2) if total else 0.0
    return {
        "total_pedestrians": total,
        "total_distracted": distracted,
        "distraction_rate_pct": rate,
        "walking_left": sum(counts[name] for name in LEFT_CLASSES),
        "walking_right": sum(counts[name] for name in RIGHT_CLASSES),
    }


def safety_badge(rate_pct: float) -> tuple[str, str, str]:
    """Map a distraction rate to a badge label, CSS class and accent colour.

    Green below 15 %, yellow from 15 % to 30 %, red above 30 %.
    """
    if rate_pct < SAFE_RATE_LIMIT:
        return "LOW RISK", "badge-green", "#16a34a"
    if rate_pct <= CAUTION_RATE_LIMIT:
        return "MODERATE", "badge-yellow", "#ca8a04"
    return "HIGH RISK", "badge-red", "#dc2626"


def class_breakdown(detections: list[Detection]) -> pd.DataFrame:
    """Per-class count, share of total and average confidence."""
    total = len(detections)
    rows: list[dict[str, Any]] = []
    for name in CLASS_NAMES:
        confs = [det["confidence"] for det in detections if det["class_name"] == name]
        rows.append(
            {
                "Class": CLASS_DISPLAY_NAMES[name],
                "Direction": "Left" if name in LEFT_CLASSES else "Right",
                "Distracted": "Yes" if name in DISTRACTED_CLASSES else "No",
                "Count": len(confs),
                "Share": (len(confs) / total) if total else 0.0,
                "Avg Confidence": float(np.mean(confs)) if confs else None,
            }
        )
    return pd.DataFrame(rows)


def build_payload(
    counts: dict[str, int],
    summary: dict[str, Any],
    confidence: float,
    image_size: tuple[int, int],
    result: InferenceResult,
) -> dict[str, Any]:
    """Build the developer API JSON payload."""
    return {
        "status": 200,
        "counts": counts,
        "summary": {
            "total_pedestrians": summary["total_pedestrians"],
            "total_distracted": summary["total_distracted"],
            "distraction_rate_pct": summary["distraction_rate_pct"],
        },
        "confidence_threshold": round(confidence, 2),
        "model_version": MODEL_VERSION,
        "image_size": [image_size[0], image_size[1]],
        "inference_seconds": round(result.inference_seconds, 4),
        "detections": result.detections,
    }


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def annotate_image(
    image: Image.Image,
    detections: list[Detection],
    settings: SidebarSettings,
) -> np.ndarray:
    """Draw class-coloured bounding boxes and optional labels.

    Args:
        image: Source RGB image.
        detections: Detections to draw.
        settings: Sidebar options (visible classes, label toggles).

    Returns:
        Annotated RGB image as a NumPy array.
    """
    canvas = np.ascontiguousarray(np.array(image, dtype=np.uint8))
    height, width = canvas.shape[:2]
    short_side = min(height, width)
    thickness = max(2, round(short_side / 320))
    font_scale = max(0.45, short_side / 1100)
    font_thickness = max(1, thickness - 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        name = det["class_name"]
        if name not in settings.visible_classes:
            continue
        color = CLASS_COLORS_RGB.get(name, FALLBACK_COLOR_RGB)
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        parts: list[str] = []
        if settings.show_labels:
            parts.append(name)
        if settings.show_confidence:
            parts.append(f"{det['confidence']:.2f}")
        if not parts:
            continue

        text = " ".join(parts)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        pad = 4
        # Put the label above the box, or inside it when the box touches the top edge.
        top = y1 - text_h - baseline - 2 * pad
        if top < 0:
            top = y1
        bottom = top + text_h + baseline + 2 * pad
        label_w = text_w + 2 * pad
        # Shift the label left when it would run past the right edge.
        left = max(0, min(x1, width - 1 - label_w))
        right = min(left + label_w, width - 1)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            canvas,
            text,
            (left + pad, bottom - baseline - pad),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas


# --------------------------------------------------------------------------- #
# UI components
# --------------------------------------------------------------------------- #

def render_sidebar() -> SidebarSettings:
    """Render the sidebar and return the chosen settings."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="title">🧠 Model Info</div>
                <div class="row"><span>Model Version</span><span>v{MODEL_VERSION}</span></div>
                <div class="row"><span>Architecture</span><span>{MODEL_ARCHITECTURE}</span></div>
                <div class="row"><span>Classes</span><span>{len(CLASS_NAMES)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Detection Settings")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            help="Detections below this confidence are discarded.",
        )
        iou = st.slider(
            "NMS / IoU Threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_IOU,
            step=0.05,
            help="Overlap threshold for non-maximum suppression. Lower values remove more overlapping boxes.",
        )

        st.subheader("Display Options")
        st.caption("Hiding a class only affects the annotated image. Counts always include every class.")
        visible: set[str] = set()
        for name in CLASS_NAMES:
            color_hex = rgb_to_hex(CLASS_COLORS_RGB[name])
            st.markdown(
                f'<span class="swatch" style="background:{color_hex}"></span>'
                f"<small>{CLASS_DISPLAY_NAMES[name]}</small>",
                unsafe_allow_html=True,
            )
            if st.checkbox(f"Show {name}", value=True, key=f"show_{name}", label_visibility="collapsed"):
                visible.add(name)

        show_labels = st.toggle("Show class labels", value=True)
        show_confidence = st.toggle("Show confidence scores", value=True)

    return SidebarSettings(
        confidence=float(confidence),
        iou=float(iou),
        visible_classes=frozenset(visible),
        show_labels=show_labels,
        show_confidence=show_confidence,
    )


def render_header() -> None:
    """Render the page title and subtitle."""
    st.markdown(
        f"""
        <div class="app-header">
            <h1>🚶 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, accent: str, badge_html: str = "") -> str:
    """Return HTML for one KPI card."""
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}{badge_html}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_kpis(summary: dict[str, Any]) -> None:
    """Render the four KPI summary cards."""
    rate = summary["distraction_rate_pct"]
    badge_label, badge_class, badge_color = safety_badge(rate)
    badge_html = f'<span class="badge {badge_class}">{badge_label}</span>'

    cards = [
        kpi_card("Total Pedestrians", str(summary["total_pedestrians"]), "All four classes", "#3478f6"),
        kpi_card("Walking Left", str(summary["walking_left"]), "person_left + phone_using_left", "#6366f1"),
        kpi_card("Walking Right", str(summary["walking_right"]), "person_right + phone_using_right", "#10ac84"),
        kpi_card(
            "Distraction Rate",
            f"{rate:.1f}%",
            f"{summary['total_distracted']} of {summary['total_pedestrians']} using a phone",
            badge_color,
            badge_html,
        ),
    ]
    for column, card in zip(st.columns(4), cards):
        column.markdown(card, unsafe_allow_html=True)


def render_breakdown(detections: list[Detection]) -> None:
    """Render the class distribution table."""
    st.markdown("#### Class Distribution")
    frame = class_breakdown(detections)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Count": st.column_config.NumberColumn("Count", format="%d"),
            "Share": st.column_config.ProgressColumn(
                "Share", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Avg Confidence": st.column_config.NumberColumn("Avg Confidence", format="%.3f"),
        },
    )
    if detections:
        st.bar_chart(frame.set_index("Class")["Count"], horizontal=True, height=220)
    else:
        st.info("No pedestrians detected at the current confidence threshold.")


def render_json_section(payload: dict[str, Any], source_name: str) -> None:
    """Render the developer JSON output with a download button."""
    with st.expander("🛠️ Developer API & Raw JSON Output", expanded=False):
        json_text = json.dumps(payload, indent=2)
        st.code(json_text, language="json")
        st.download_button(
            label="⬇️ Download JSON",
            data=json_text.encode("utf-8"),
            file_name=f"{Path(source_name).stem}_detections.json",
            mime="application/json",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title=f"{APP_TITLE} | Distracted Pedestrian Detection",
        page_icon="🚶",
        layout="wide",
    )
    inject_css()
    settings = render_sidebar()
    render_header()

    try:
        model = load_model(str(MODEL_PATH))
    except FileNotFoundError:
        st.error(
            f"Model weights were not found at `models/best.pt`. "
            f"Copy your trained `best.pt` into `{MODEL_PATH.parent}` and reload the page."
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        st.error(f"The model could not be loaded: {exc}")
        st.stop()

    loaded_names = model_class_names(model)
    if loaded_names != list(CLASS_NAMES):
        st.warning(
            f"The model's classes {loaded_names} do not match the expected classes "
            f"{list(CLASS_NAMES)}. Counts may be wrong."
        )

    uploaded = st.file_uploader(
        "Upload a walkway image",
        type=list(ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        help=f"JPG, JPEG or PNG, up to {MAX_UPLOAD_BYTES // 1_048_576} MB.",
    )
    if uploaded is None:
        st.info("Upload a JPG, JPEG or PNG image of a walkway to start.")
        return

    raw = uploaded.getvalue()
    try:
        image = load_uploaded_image(uploaded.name, raw)
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    digest = hashlib.sha256(raw).hexdigest()
    try:
        with st.spinner("Running detection…"):
            detections, seconds = run_inference(
                digest, image, model, settings.confidence, settings.iou, MODEL_VERSION
            )
        result = InferenceResult(detections=detections, inference_seconds=seconds)
    except Exception as exc:  # noqa: BLE001 - inference errors should not crash the page
        st.error(f"Inference failed: {exc}")
        return

    counts = compute_counts(result.detections)
    summary = compute_summary(counts)

    render_kpis(summary)
    st.write("")

    image_col, stats_col = st.columns([3, 2], gap="large")
    with image_col:
        st.markdown("#### Annotated Image")
        annotated = annotate_image(image, result.detections, settings)
        st.image(
            annotated,
            caption=f"{uploaded.name} · {image.width}×{image.height}px",
            width="stretch",
        )
        st.markdown(
            f'<span class="timing-pill">⏱️ Inference time: {result.inference_seconds:.3f} s</span>',
            unsafe_allow_html=True,
        )
    with stats_col:
        render_breakdown(result.detections)

    st.write("")
    payload = build_payload(counts, summary, settings.confidence, image.size, result)
    render_json_section(payload, uploaded.name)


if __name__ == "__main__":
    main()
