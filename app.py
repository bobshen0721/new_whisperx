import argparse
import gc
import html
import json
import os
import string
import uuid
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import gradio as gr
import numpy as np
import torch
import torchaudio

import whisperx
from whisperx.diarize import DiarizationPipeline


REPO_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = REPO_ROOT / "models"
OUTPUT_ROOT = REPO_ROOT / "gradio_outputs"

MAX_AUDIO_SECONDS = 2 * 60 * 60
DEFAULT_CHUNK_SIZE = 6
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_COMPUTE_TYPE = "float16" if DEFAULT_DEVICE == "cuda" else "int8"
DEFAULT_BATCH_SIZE = 2 if DEFAULT_DEVICE == "cuda" else 1
UNKNOWN_CONFIDENCE_THRESHOLD = 0.75
UNKNOWN_MARGIN_THRESHOLD = 0.30
APP_CONFIG = {
    "model_choice": os.getenv("WHISPERX_WEBUI_MODEL", "auto"),
}
DEFAULT_HF_TOKEN = os.getenv("WHISPERX_HF_TOKEN", "").strip()
LANGUAGE_CHOICES = [
    ("自動", "auto"),
    ("繁體中文", "zh"),
    ("English", "en"),
]
SPEAKER_COUNT_CHOICES = [("自動推斷", "auto")] + [
    (f"{count} 位", str(count)) for count in range(1, 9)
]

CLOSING_PUNCTUATION = set(".,!?;:)]}%>\"'，。！？；：、）》」』】")
OPENING_PUNCTUATION = set("([<{\"'《「『【")
TEXT_PUNCTUATION = CLOSING_PUNCTUATION | OPENING_PUNCTUATION | set(string.punctuation)

SPEAKER_THEMES = [
    {"bg": "#E8F1FF", "accent": "#1D4ED8"},
    {"bg": "#FFF1E8", "accent": "#C2410C"},
    {"bg": "#EAFBF0", "accent": "#15803D"},
    {"bg": "#FFF7D6", "accent": "#A16207"},
    {"bg": "#F3E8FF", "accent": "#7E22CE"},
    {"bg": "#FDE7EF", "accent": "#BE185D"},
    {"bg": "#E6FCF5", "accent": "#0F766E"},
    {"bg": "#F3F4F6", "accent": "#4B5563"},
]
UNKNOWN_THEME = {"bg": "#F3F4F6", "accent": "#6B7280"}

CUSTOM_CSS = """
.app-shell {
  max-width: 1360px;
  margin: 0 auto;
}
.app-note {
  color: #475569;
}
.workspace-grid {
  display: grid;
  grid-template-columns: minmax(320px, 0.92fr) minmax(520px, 1.35fr);
  gap: 22px;
  align-items: start;
}
.left-pane,
.right-pane {
  min-width: 0;
}
.control-card,
.transcript-panel {
  border-radius: 22px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}
.control-card {
  padding: 18px;
}
.player-panel {
  margin: 14px 0 16px;
  border-radius: 18px;
  border: 1px solid #dbe2ea;
  background: linear-gradient(180deg, #f8fafc, #eef2f7);
  padding: 14px;
}
.player-panel h3 {
  margin: 0 0 6px;
  font-size: 0.98rem;
}
.player-note {
  margin: 0 0 10px;
  color: #64748b;
  font-size: 0.85rem;
}
.player-panel audio {
  width: 100%;
}
.transcript-panel {
  padding: 18px 18px 10px;
}
.transcript-panel h3 {
  margin: 0 0 8px;
  font-size: 1.05rem;
}
.transcript-scroller {
  max-height: 76vh;
  overflow: auto;
  padding-right: 6px;
}
.speaker-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.speaker-block {
  background: var(--block-bg);
  border: 1px solid color-mix(in srgb, var(--block-accent) 18%, white);
  border-left: 5px solid var(--block-accent);
  border-radius: 14px;
  padding: 10px 12px 11px;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
  cursor: pointer;
  transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease;
}
.speaker-block:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px rgba(15, 23, 42, 0.08);
  border-color: color-mix(in srgb, var(--block-accent) 40%, white);
}
.speaker-block:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--block-accent) 22%, white),
    0 12px 22px rgba(15, 23, 42, 0.08);
  border-color: color-mix(in srgb, var(--block-accent) 55%, white);
}
.speaker-block.is-active {
  transform: translateY(-1px);
  border-color: color-mix(in srgb, var(--block-accent) 58%, white);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--block-accent) 18%, white),
    0 16px 28px rgba(15, 23, 42, 0.12);
}
.speaker-block.is-active .seek-chip {
  background: var(--block-accent);
  color: #ffffff;
  box-shadow: none;
}
.speaker-block.is-active .speaker-text {
  color: color-mix(in srgb, #0f172a 90%, var(--block-accent));
}
.speaker-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 5px;
}
.speaker-name {
  color: var(--block-accent);
  font-weight: 700;
  font-size: 0.88rem;
  letter-spacing: 0.01em;
  text-transform: uppercase;
}
.speaker-time {
  color: #475569;
  font-size: 0.88rem;
  font-variant-numeric: tabular-nums;
}
.seek-chip {
  appearance: none;
  border: 0;
  border-radius: 999px;
  padding: 0.22rem 0.58rem;
  background: rgba(255, 255, 255, 0.92);
  color: #0f172a;
  cursor: pointer;
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
}
.seek-chip:hover {
  background: #ffffff;
}
.speaker-text {
  color: #0f172a;
  line-height: 1.48;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.97rem;
}
.transcript-help {
  margin-bottom: 12px;
  color: #475569;
  font-size: 0.88rem;
}
.summary-card {
  background: linear-gradient(145deg, #f8fafc, #eef2ff);
  border: 1px solid #dbeafe;
  border-radius: 18px;
  padding: 18px;
}
.summary-card ul {
  margin: 10px 0 0 0;
  padding-left: 18px;
}
@media (max-width: 980px) {
  .workspace-grid {
    grid-template-columns: 1fr;
  }
  .transcript-scroller {
    max-height: none;
  }
}
"""

APP_SCRIPT = """
(() => {
  const TRANSCRIPT_BLOCK_SELECTOR = ".speaker-block[data-start-seconds][data-end-seconds]";
  let activeBlock = null;

  const findAudioPlayer = () => document.querySelector('#native-audio-player');
  const getTranscriptBlocks = () => Array.from(document.querySelectorAll(TRANSCRIPT_BLOCK_SELECTOR));

  const setActiveBlock = (block) => {
    if (activeBlock === block) return;
    if (activeBlock && activeBlock.isConnected) {
      activeBlock.classList.remove("is-active");
      activeBlock.setAttribute("aria-current", "false");
    }
    activeBlock = block && block.isConnected ? block : null;
    if (activeBlock) {
      activeBlock.classList.add("is-active");
      activeBlock.setAttribute("aria-current", "true");
    }
  };

  const findBlockForTime = (seconds) => {
    if (!Number.isFinite(seconds)) return null;
    const blocks = getTranscriptBlocks();
    if (!blocks.length) return null;
    const epsilon = 0.08;
    for (const block of blocks) {
      const start = Number.parseFloat(block.dataset.startSeconds || "");
      const end = Number.parseFloat(block.dataset.endSeconds || "");
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      if (seconds >= start - epsilon && seconds < end + epsilon) {
        return block;
      }
    }
    return null;
  };

  const syncActiveBlockFromPlayer = () => {
    const player = findAudioPlayer();
    if (!player) {
      setActiveBlock(null);
      return;
    }
    if (player.paused && player.currentTime <= 0.05) {
      return;
    }
    setActiveBlock(findBlockForTime(player.currentTime));
  };

  const seekToTime = (element) => {
    const seconds = Number.parseFloat(element.dataset.seekSeconds || "");
    if (Number.isNaN(seconds)) return;
    const player = findAudioPlayer();
    if (!player) return;
    setActiveBlock(element.closest(".speaker-block") || findBlockForTime(seconds));
    player.currentTime = seconds;
    const playPromise = player.play?.();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
  };

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-seek-seconds]");
    if (!target) return;
    event.preventDefault();
    seekToTime(target);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = event.target.closest("[data-seek-seconds]");
    if (!target) return;
    event.preventDefault();
    seekToTime(target);
  });

  window.setInterval(syncActiveBlockFromPlayer, 200);
})();
"""

HEAD_SCRIPT = f"<script>{APP_SCRIPT}</script>"


def resolve_whisper_model(model_choice: str | None = None) -> tuple[str, str]:
    choice = (model_choice or APP_CONFIG["model_choice"] or "auto").strip()
    if choice and choice.lower() != "auto":
        if Path(choice).exists():
            return choice, f"{Path(choice).name}（本機模型路徑）"
        return choice, f"{choice}（指定模型）"

    local_model_dir = MODELS_ROOT / "faster-whisper-large-v2"
    if (local_model_dir / "model.bin").exists():
        return str(local_model_dir), "large-v2（本機模型包）"
    return "large-v2", "large-v2（首次執行可能會下載模型）"


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_timestamp(seconds: float, force_hours: bool = False) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    total_seconds, _ = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if force_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    total_seconds, millis = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_duration(seconds: float) -> str:
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    if hours:
        return f"{hours} 小時 {minutes} 分 {secs} 秒"
    return f"{minutes} 分 {secs} 秒"


def normalize_language_choice(choice: str | None) -> str | None:
    if not choice or choice == "auto":
        return None
    return choice


def normalize_speaker_count_choice(choice: str | None) -> int | None:
    if not choice or choice == "auto":
        return None
    try:
        value = int(choice)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def format_language_label(language_code: str | None) -> str:
    if language_code == "zh":
        return "繁體中文"
    if language_code == "en":
        return "English"
    if language_code:
        return language_code
    return "自動"


def get_audio_duration_seconds(audio_path: str) -> float:
    try:
        metadata = torchaudio.info(audio_path)
        if metadata.sample_rate > 0 and metadata.num_frames > 0:
            return metadata.num_frames / metadata.sample_rate
    except Exception:
        pass

    waveform = load_audio_array(audio_path)
    return float(len(waveform) / 16000.0)


def load_audio_array(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            original_sample_rate,
            sample_rate,
        )
    return waveform.squeeze(0).numpy().astype(np.float32)


def validate_audio(audio_path: str) -> float:
    if not audio_path:
        raise gr.Error("請先上傳音檔。")

    path = Path(audio_path)
    if not path.exists():
        raise gr.Error("找不到上傳的暫存音檔，請重新上傳一次。")

    duration = get_audio_duration_seconds(str(path))
    if duration > MAX_AUDIO_SECONDS:
        raise gr.Error(
            f"這個檔案約 {format_duration(duration)}，超過目前 2 小時上限。"
        )
    return duration


def build_audio_player_html(audio_path: str | None) -> str:
    if not audio_path:
        return (
            "<div class='player-panel'>"
            "<h3>播放器</h3>"
            "<p class='player-note'>先上傳音檔，這裡會出現可跳播的原生播放器。</p>"
            "</div>"
        )

    path = Path(audio_path)
    resolved = str(path.resolve()) if path.exists() else str(path)
    src = "/gradio_api/file=" + quote(resolved.replace("\\", "/"), safe="/:=._-() ")
    return (
        "<div class='player-panel'>"
        "<h3>播放器</h3>"
        "<p class='player-note'>逐字稿點擊後會直接控制這個播放器。</p>"
        f"<audio id='native-audio-player' controls preload='metadata' src='{html.escape(src, quote=True)}'></audio>"
        "</div>"
    )


def has_cjk(text: str) -> bool:
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            return True
    return False


def should_insert_space(previous_token: str, token: str) -> bool:
    if not previous_token or not token:
        return False
    if token[0] in CLOSING_PUNCTUATION:
        return False
    if previous_token[-1] in OPENING_PUNCTUATION:
        return False
    if has_cjk(previous_token) or has_cjk(token):
        return False
    return True


def render_tokens(tokens: Iterable[str]) -> str:
    text = ""
    previous_token = ""
    for raw_token in tokens:
        token = str(raw_token).strip()
        if not token:
            continue
        if not text:
            text = token
        elif should_insert_space(previous_token, token):
            text += " " + token
        else:
            text += token
        previous_token = token
    return text


def block_duration(block: dict) -> float:
    return max(float(block["end"]) - float(block["start"]), 0.0)


def block_visible_char_count(block: dict) -> int:
    return sum(
        1
        for char in str(block.get("text", ""))
        if not char.isspace() and char not in TEXT_PUNCTUATION
    )


def flatten_word_speaker_segments(result: dict) -> list[dict]:
    words = []
    for segment in result.get("segments", []):
        default_speaker = segment.get("speaker") or "UNKNOWN"
        for word in segment.get("words", []):
            token = str(word.get("word", "")).strip()
            if not token or "start" not in word:
                continue
            start = float(word["start"])
            end = float(word.get("end", start))
            words.append(
                {
                    "word": token,
                    "start": start,
                    "end": end,
                    "speaker": word.get("speaker") or default_speaker,
                }
            )
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def regroup_by_word_speaker(result: dict, gap_threshold: float = 0.8) -> list[dict]:
    words = flatten_word_speaker_segments(result)
    if not words:
        return []

    blocks = []
    current = None
    for word in words:
        if current is None:
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word],
            }
            continue

        gap = word["start"] - current["end"]
        if word["speaker"] != current["speaker"] or gap > gap_threshold:
            blocks.append(current)
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word],
            }
            continue

        current["words"].append(word)
        current["end"] = max(current["end"], word["end"])

    if current is not None:
        blocks.append(current)

    regrouped = []
    for block in blocks:
        regrouped.append(
            {
                "start": round(block["start"], 3),
                "end": round(block["end"], 3),
                "speaker": block["speaker"],
                "text": render_tokens(item["word"] for item in block["words"]),
                "words": list(block["words"]),
            }
        )
    return regrouped


def smooth_micro_speaker_turns(blocks: list[dict]) -> list[dict]:
    if len(blocks) < 3:
        return blocks

    smoothed = [dict(block) for block in blocks]
    changed = True
    while changed:
        changed = False
        for index in range(1, len(smoothed) - 1):
            previous = smoothed[index - 1]
            current = smoothed[index]
            following = smoothed[index + 1]
            is_micro = (
                block_duration(current) <= 0.45
                or block_visible_char_count(current) <= 2
                or len(current.get("words", [])) <= 2
            )
            if (
                is_micro
                and previous.get("speaker")
                and previous.get("speaker") == following.get("speaker")
                and previous.get("speaker") != current.get("speaker")
            ):
                current["speaker"] = previous["speaker"]
                for word in current.get("words", []):
                    word["speaker"] = previous["speaker"]
                changed = True
        if changed:
            smoothed = merge_adjacent_same_speaker(smoothed, gap_threshold=0.3)
    return smoothed


def merge_adjacent_same_speaker(blocks: list[dict], gap_threshold: float = 0.3) -> list[dict]:
    if not blocks:
        return []

    merged = [dict(blocks[0])]
    for block in blocks[1:]:
        previous = merged[-1]
        gap = float(block["start"]) - float(previous["end"])
        if block.get("speaker") == previous.get("speaker") and gap <= gap_threshold:
            previous["end"] = round(max(float(previous["end"]), float(block["end"])), 3)
            previous_words = list(previous.get("words", []))
            current_words = list(block.get("words", []))
            previous["words"] = previous_words + current_words
            previous["text"] = render_tokens(
                item["word"] for item in previous["words"]
            ) or f"{previous.get('text', '').strip()} {block.get('text', '').strip()}".strip()
        else:
            merged.append(dict(block))
    return merged


def speaker_confidence_stats(block: dict) -> tuple[str | None, float, float]:
    durations: dict[str, float] = {}
    total = 0.0

    for word in block.get("words", []):
        speaker = str(word.get("speaker") or "").strip()
        if not speaker:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        duration = max(end - start, 0.0)
        if duration == 0:
            duration = 0.01
        durations[speaker] = durations.get(speaker, 0.0) + duration
        total += duration

    if not durations or total <= 0:
        return None, 0.0, 0.0

    ranked = sorted(durations.items(), key=lambda item: item[1], reverse=True)
    top_speaker, top_duration = ranked[0]
    second_duration = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = top_duration / total
    margin = (top_duration - second_duration) / total
    return top_speaker, round(confidence, 3), round(margin, 3)


def mark_uncertain_speakers_unknown(
    blocks: list[dict],
    confidence_threshold: float = UNKNOWN_CONFIDENCE_THRESHOLD,
    margin_threshold: float = UNKNOWN_MARGIN_THRESHOLD,
) -> list[dict]:
    updated = []
    for block in blocks:
        normalized = dict(block)
        dominant_speaker, confidence, margin = speaker_confidence_stats(normalized)
        normalized["speaker_confidence"] = confidence
        normalized["speaker_margin"] = margin

        if (
            dominant_speaker is None
            or confidence < confidence_threshold
            or margin < margin_threshold
        ):
            normalized["speaker"] = "UNKNOWN"
        else:
            normalized["speaker"] = dominant_speaker
        updated.append(normalized)
    return updated


def has_word_speaker_annotations(result: dict) -> bool:
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if word.get("speaker"):
                return True
    return False


def build_display_rows(result: dict, prefer_word_speakers: bool) -> tuple[list[dict], dict[str, str]]:
    segments = []
    if prefer_word_speakers and has_word_speaker_annotations(result):
        segments = regroup_by_word_speaker(result)
        segments = smooth_micro_speaker_turns(segments)
        segments = merge_adjacent_same_speaker(segments)
        segments = mark_uncertain_speakers_unknown(segments)

    if not segments:
        segments = []
        for segment in result.get("segments", []):
            fallback = dict(segment)
            fallback["speaker"] = "UNKNOWN"
            segments.append(fallback)

    speaker_aliases: dict[str, str] = {}
    next_index = 0
    rows = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        raw_speaker = str(segment.get("speaker") or "UNKNOWN").strip()
        if raw_speaker and raw_speaker != "UNKNOWN":
            if raw_speaker not in speaker_aliases:
                speaker_aliases[raw_speaker] = f"講者{speaker_letter(next_index)}"
                next_index += 1
            speaker_name = speaker_aliases[raw_speaker]
        else:
            speaker_name = "UNKNOWN"

        rows.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "raw_speaker": raw_speaker,
                "speaker": speaker_name,
                "text": text,
                "speaker_confidence": float(segment.get("speaker_confidence", 0.0)),
                "speaker_margin": float(segment.get("speaker_margin", 0.0)),
            }
        )

    return rows, speaker_aliases


def speaker_letter(index: int) -> str:
    letters = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(letters))


def build_speaker_theme_map(rows: list[dict]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    theme_index = 0
    for row in rows:
        speaker = row["speaker"]
        if speaker == "UNKNOWN":
            mapping[speaker] = UNKNOWN_THEME
            continue
        if speaker not in mapping:
            mapping[speaker] = SPEAKER_THEMES[theme_index % len(SPEAKER_THEMES)]
            theme_index += 1
    if "UNKNOWN" not in mapping:
        mapping["UNKNOWN"] = UNKNOWN_THEME
    return mapping


def render_transcript_html(rows: list[dict], total_duration: float) -> str:
    if not rows:
        return "<div class='speaker-list'><p class='app-note'>結果會顯示在這裡。</p></div>"

    force_hours = total_duration >= 3600
    themes = build_speaker_theme_map(rows)
    blocks = []
    for row in rows:
        theme = themes[row["speaker"]]
        blocks.append(
            (
                "<article class='speaker-block' "
                f"style='--block-bg:{theme['bg']};--block-accent:{theme['accent']}' "
                f"data-seek-seconds='{row['start']:.3f}' "
                f"data-start-seconds='{row['start']:.3f}' "
                f"data-end-seconds='{row['end']:.3f}' "
                "tabindex='0' role='button' aria-current='false'>"
                "<div class='speaker-meta'>"
                f"<span class='speaker-name'>{html.escape(row['speaker'])}</span>"
                f"<button type='button' class='seek-chip' data-seek-seconds='{row['start']:.3f}'>"
                f"{format_timestamp(row['start'], force_hours)} - {format_timestamp(row['end'], force_hours)}</button>"
                "</div>"
                f"<div class='speaker-text'>{html.escape(row['text'])}</div>"
                "</article>"
            )
        )
    return (
        "<div class='transcript-panel'>"
        "<h3>逐字稿</h3>"
        "<div class='transcript-help'>點任一行或時間標籤，左側播放器會跳到該段開始播放。</div>"
        "<div class='transcript-scroller'><div class='speaker-list'>"
        + "".join(blocks)
        + "</div></div></div>"
    )


def render_plain_text(rows: list[dict], total_duration: float) -> str:
    if not rows:
        return ""
    force_hours = total_duration >= 3600
    return "\n".join(
        f"[{format_timestamp(row['start'], force_hours)}-{format_timestamp(row['end'], force_hours)}] "
        f"{row['speaker']}: {row['text']}"
        for row in rows
    )


def chunked_stream(rows: list[dict], total_updates: int = 24) -> Iterable[list[dict]]:
    if not rows:
        return []

    step = max(len(rows) // total_updates, 1)
    output = []
    for index in range(step, len(rows) + step, step):
        output.append(rows[: min(index, len(rows))])
    if output[-1] != rows:
        output.append(rows)
    return output


def build_summary_markdown(
    *,
    source_name: str,
    duration: float,
    language: str | None,
    speaker_aliases: dict[str, str],
) -> str:
    speaker_count = len(speaker_aliases)
    detected_language = format_language_label(language)
    lines = [
        "<div class='summary-card'>",
        "<h3>處理結果</h3>",
        "<ul>",
        f"<li><strong>檔名：</strong>{html.escape(source_name)}</li>",
        f"<li><strong>音檔長度：</strong>{html.escape(format_duration(duration))}</li>",
        f"<li><strong>辨識語言：</strong>{html.escape(detected_language)}</li>",
        f"<li><strong>講者數：</strong>{speaker_count if speaker_count else '未辨識'}</li>",
        "</ul>",
        "</div>",
    ]
    return "\n".join(lines)


def create_job_dir(source_name: str) -> Path:
    job_dir = OUTPUT_ROOT / f"{Path(source_name).stem}_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def write_outputs(
    *,
    job_dir: Path,
    source_name: str,
    duration: float,
    language: str | None,
    rows: list[dict],
    speaker_aliases: dict[str, str],
) -> list[str]:
    force_hours = duration >= 3600
    safe_stem = Path(source_name).stem

    txt_path = job_dir / f"{safe_stem}.txt"
    srt_path = job_dir / f"{safe_stem}.srt"
    json_path = job_dir / f"{safe_stem}.json"
    html_path = job_dir / f"{safe_stem}.html"
    zip_path = job_dir / f"{safe_stem}_bundle.zip"

    plain_text = render_plain_text(rows, duration)
    txt_path.write_text(plain_text + "\n", encoding="utf-8")

    srt_lines = []
    for index, row in enumerate(rows, start=1):
        srt_lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(row['start'])} --> {format_srt_timestamp(row['end'])}",
                f"{row['speaker']}：{row['text']}",
                "",
            ]
        )
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    export_json = {
        "source_file": source_name,
        "duration_seconds": round(duration, 3),
        "language": language,
        "speaker_aliases": speaker_aliases,
        "segments": [
            {
                "start": round(row["start"], 3),
                "end": round(row["end"], 3),
                "speaker": row["speaker"],
                "raw_speaker": row["raw_speaker"],
                "text": row["text"],
                "start_label": format_timestamp(row["start"], force_hours),
                "end_label": format_timestamp(row["end"], force_hours),
            }
            for row in rows
        ],
    }
    json_path.write_text(
        json.dumps(export_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    standalone_html = (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>WhisperX Transcript</title>"
        f"<style>{CUSTOM_CSS}</style></head><body>"
        f"{render_transcript_html(rows, duration)}</body></html>"
    )
    html_path.write_text(standalone_html, encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(txt_path, arcname=txt_path.name)
        archive.write(srt_path, arcname=srt_path.name)
        archive.write(json_path, arcname=json_path.name)
        archive.write(html_path, arcname=html_path.name)

    return [str(zip_path), str(txt_path), str(srt_path), str(json_path), str(html_path)]


def run_transcription(
    audio_path: str,
    language_choice: str,
    speaker_count_choice: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    duration = validate_audio(audio_path)
    source_path = Path(audio_path)
    model_spec, _model_label = resolve_whisper_model()
    token = DEFAULT_HF_TOKEN
    selected_language = normalize_language_choice(language_choice)
    selected_speaker_count = normalize_speaker_count_choice(speaker_count_choice)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = create_job_dir(source_path.name)

    progress(0.02, desc="載入 WhisperX")
    asr_model = whisperx.load_model(
        model_spec,
        DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE_TYPE,
        download_root=str(MODELS_ROOT),
        language=selected_language,
        vad_method="pyannote",
        vad_options={"chunk_size": DEFAULT_CHUNK_SIZE},
        use_auth_token=token or None,
    )
    audio = load_audio_array(str(source_path))

    progress(0.08, desc="語音轉錄中")
    asr_result = asr_model.transcribe(
        audio,
        batch_size=DEFAULT_BATCH_SIZE,
        language=selected_language,
        chunk_size=DEFAULT_CHUNK_SIZE,
        progress_callback=lambda pct: progress(
            0.08 + (pct / 100.0) * 0.42, desc="語音轉錄中"
        ),
    )

    del asr_model
    cleanup_memory()

    progress(0.55, desc="文字時間對齊中")
    align_model, align_metadata = whisperx.load_align_model(
        asr_result.get("language") or "en",
        DEFAULT_DEVICE,
        model_dir=str(MODELS_ROOT),
    )
    aligned_result = whisperx.align(
        asr_result["segments"],
        align_model,
        align_metadata,
        audio,
        DEFAULT_DEVICE,
        progress_callback=lambda pct: progress(
            0.55 + (pct / 100.0) * 0.20, desc="文字時間對齊中"
        ),
    )
    aligned_result["language"] = asr_result.get("language")

    del align_model
    cleanup_memory()

    final_result = aligned_result
    if token:
        try:
            progress(0.78, desc="講者辨識中")
            diarize_model = DiarizationPipeline(
                token=token,
                device=DEFAULT_DEVICE,
                cache_dir=str(MODELS_ROOT),
            )
            diarize_segments = diarize_model(
                audio,
                num_speakers=selected_speaker_count,
                progress_callback=lambda pct: progress(
                    0.78 + (pct / 100.0) * 0.18, desc="講者辨識中"
                ),
            )
            final_result = whisperx.assign_word_speakers(
                diarize_segments,
                final_result,
            )
            del diarize_model
            cleanup_memory()
        except Exception as exc:
            note = (
                "已完成轉錄與對齊，但 diarization 啟動失敗，speaker 退回 UNKNOWN。"
                f" 原因：{exc}"
            )

    final_rows, speaker_aliases = build_display_rows(
        final_result,
        prefer_word_speakers=bool(token),
    )
    downloads = write_outputs(
        job_dir=job_dir,
        source_name=source_path.name,
        duration=duration,
        language=final_result.get("language"),
        rows=final_rows,
        speaker_aliases=speaker_aliases,
    )

    progress(1.0, desc="完成")
    return (
        build_summary_markdown(
            source_name=source_path.name,
            duration=duration,
            language=final_result.get("language"),
            speaker_aliases=speaker_aliases,
        ),
        render_transcript_html(final_rows, duration),
        downloads,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="語音轉錄平台2.0v",
        theme=gr.themes.Glass(),
        css=CUSTOM_CSS,
        head=HEAD_SCRIPT,
        fill_height=True,
    ) as demo:
        gr.Markdown(
            """
            <div class="app-shell">
              <h1>語音轉錄平台2.0v</h1>
            </div>
            """
        )

        with gr.Row(elem_classes=["workspace-grid"]):
            with gr.Column(elem_classes=["left-pane"]):
                with gr.Group(elem_classes=["control-card"]):
                    audio_input = gr.File(
                        label="上傳音檔",
                        type="filepath",
                        file_types=[".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"],
                    )
                    player_html = gr.HTML(
                        value=build_audio_player_html(None),
                        elem_id="source-audio-player",
                    )
                    language_choice = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value="auto",
                        label="轉錄語言",
                        info="可選自動、繁體中文或 English；預設為自動偵測。",
                    )
                    speaker_count_choice = gr.Dropdown(
                        choices=SPEAKER_COUNT_CHOICES,
                        value="auto",
                        label="講者人數",
                        info="預設自動推斷；若你已知是 2 位、4 位，也可以固定指定。",
                    )
                    run_button = gr.Button("開始轉錄", variant="primary", size="lg")
                    summary_output = gr.HTML(label="摘要")
                    downloads = gr.File(
                        label="下載結果",
                        file_count="multiple",
                    )
            with gr.Column(elem_classes=["right-pane"]):
                transcript_html = gr.HTML(label="彩色講者區塊", elem_id="transcript-output")

        audio_input.change(
            fn=build_audio_player_html,
            inputs=audio_input,
            outputs=player_html,
            show_progress="hidden",
        )

        run_button.click(
            fn=run_transcription,
            inputs=[audio_input, language_choice, speaker_count_choice],
            outputs=[summary_output, transcript_html, downloads],
            show_progress="minimal",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the WhisperX Gradio web UI.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--model",
        default=APP_CONFIG["model_choice"],
        help="Whisper model name or local model path. Use 'auto' to prefer the bundled large-v2 package.",
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    APP_CONFIG["model_choice"] = args.model
    demo = build_demo()
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=args.share,
        allowed_paths=[str(OUTPUT_ROOT)],
        max_file_size="2gb",
        show_api=False,
    )


if __name__ == "__main__":
    main()
