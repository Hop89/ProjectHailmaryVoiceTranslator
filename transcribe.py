import argparse
import wave
from pathlib import Path

import pyaudio


def record_continuously(output_path, sample_rate=16000, chunk_size=1024):
    """Continuously record microphone audio into a WAV file until stopped."""
    audio_format = pyaudio.paInt16
    channels = 1
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=audio_format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    wav_file = wave.open(str(output_file), "wb")
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(audio.get_sample_size(audio_format))
    wav_file.setframerate(sample_rate)

    print(f"Recording to {output_file}. Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            wav_file.writeframes(data)
            # Keep the WAV header sizes in sync so the file remains readable.
            wav_file._patchheader()
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        wav_file.close()
        stream.stop_stream()
        stream.close()
        audio.terminate()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuously record microphone audio into a WAV file."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="recordings/live_capture.wav",
        help="Path to the WAV file to keep updating.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Number of frames per read from the microphone.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_continuously(
        output_path=args.output,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )
