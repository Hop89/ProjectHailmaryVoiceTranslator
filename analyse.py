import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from translate import translate_notes_to_english


A4_FREQUENCY = 440.0
A4_MIDI = 69
MIN_PIANO_FREQUENCY = 27.5
MAX_PIANO_FREQUENCY = 4186.01
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class NoteEvent:
    note: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a WAV recording and extract distinct piano notes."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="recordings/live_capture.wav",
        help="Path to the WAV file produced by transcribe.py.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=0.08,
        help="Analysis window size in seconds.",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=0.02,
        help="Time between analysis frames in seconds.",
    )
    parser.add_argument(
        "--min-note-duration",
        type=float,
        default=0.06,
        help="Minimum duration in seconds for a note event to be kept.",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=0.01,
        help="Minimum normalized frame loudness required to detect a note.",
    )
    return parser.parse_args()


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channel_count = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width == 1:
        samples = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width * 8} bits")

    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)

    return sample_rate, samples


def frequency_to_note_name(frequency: float) -> str | None:
    if frequency <= 0:
        return None

    midi = round(12 * math.log2(frequency / A4_FREQUENCY) + A4_MIDI)
    if midi < 21 or midi > 108:
        return None

    octave = (midi // 12) - 1
    note_name = NOTE_NAMES[midi % 12]
    return f"{note_name}{octave}"


def detect_frame_note(
    frame: np.ndarray,
    sample_rate: int,
    min_rms: float,
) -> str | None:
    rms = float(np.sqrt(np.mean(frame * frame)))
    if rms < min_rms:
        return None

    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)

    valid = (freqs >= MIN_PIANO_FREQUENCY) & (freqs <= MAX_PIANO_FREQUENCY)
    if not np.any(valid):
        return None

    usable_spectrum = spectrum[valid]
    usable_freqs = freqs[valid]
    if usable_spectrum.size < 8:
        return None

    hps = usable_spectrum.copy()
    for factor in range(2, 5):
        downsampled = usable_spectrum[::factor]
        hps[: downsampled.size] *= downsampled

    peak_index = int(np.argmax(hps))
    peak_strength = float(hps[peak_index])
    baseline = float(np.mean(hps) + 1e-12)
    if peak_strength < baseline * 6:
        return None

    return frequency_to_note_name(float(usable_freqs[peak_index]))


def detect_note_events(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    min_note_duration: float,
    min_rms: float,
) -> list[NoteEvent]:
    window_size = max(256, int(window_seconds * sample_rate))
    hop_size = max(1, int(hop_seconds * sample_rate))
    if len(samples) < window_size:
        return []

    frame_notes: list[tuple[float, str | None]] = []
    for start in range(0, len(samples) - window_size + 1, hop_size):
        frame = samples[start : start + window_size]
        frame_time = start / sample_rate
        note = detect_frame_note(frame, sample_rate, min_rms=min_rms)
        frame_notes.append((frame_time, note))

    events: list[NoteEvent] = []
    active_note = None
    active_start = 0.0
    active_end = 0.0

    for frame_time, note in frame_notes:
        frame_end = frame_time + window_size / sample_rate
        if note == active_note:
            active_end = frame_end
            continue

        if active_note is not None:
            event = NoteEvent(note=active_note, start=active_start, end=active_end)
            if event.duration >= min_note_duration:
                events.append(event)

        active_note = note
        active_start = frame_time
        active_end = frame_end

    if active_note is not None:
        event = NoteEvent(note=active_note, start=active_start, end=active_end)
        if event.duration >= min_note_duration:
            events.append(event)

    return events


def distinct_notes(events: list[NoteEvent]) -> list[str]:
    ordered_notes: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.note not in seen:
            seen.add(event.note)
            ordered_notes.append(event.note)
    return ordered_notes


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"WAV file not found: {input_path}")

    sample_rate, samples = read_wav_mono(input_path)
    events = detect_note_events(
        samples=samples,
        sample_rate=sample_rate,
        window_seconds=args.window_size,
        hop_seconds=args.hop_size,
        min_note_duration=args.min_note_duration,
        min_rms=args.min_rms,
    )
    notes = distinct_notes(events)

    print(f"Input: {input_path}")
    if not notes:
        print("No distinct notes were detected.")
        return

    english_translation = translate_notes_to_english(notes)
    print(english_translation)
    print("\nDetected note events:")
    for event in events:
        print(f"{event.start:7.2f}s - {event.end:7.2f}s  {event.note}")


if __name__ == "__main__":
    main()
