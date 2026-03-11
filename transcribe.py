import platform

import numpy as np
import pyaudio
import whisper


# Keep the default model light so local transcription stays responsive on Windows.
model = whisper.load_model("tiny.en")


def transcribe_directly(duration_seconds=3, sample_rate=16000):
    """Record a short microphone clip and transcribe it with Whisper."""
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        print(
            f"Recording for {duration_seconds} seconds on {platform.system()}..."
        )

        frames = []
        total_chunks = int(sample_rate / chunk_size * duration_seconds)
        for _ in range(total_chunks):
            frames.append(stream.read(chunk_size, exception_on_overflow=False))
    finally:
        if "stream" in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()

    audio_bytes = b"".join(frames)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_array /= 32768.0

    result = model.transcribe(audio_array, fp16=False, language="en")
    return result["text"].strip()


if __name__ == "__main__":
    print(transcribe_directly())
