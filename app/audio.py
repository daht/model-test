from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
import re

from fastapi import HTTPException, UploadFile, status


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def validate_audio_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing audio filename",
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio file type",
        )
    return suffix


async def save_upload_to_tempfile(upload: UploadFile, max_bytes: int) -> str:
    suffix = validate_audio_filename(upload.filename)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(upload.filename or "audio").stem)
    prefix = f"{stem[:48]}-"
    total = 0

    with tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix) as output:
        temp_path = output.name
        try:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Audio file is too large",
                    )
                output.write(chunk)
        except Exception:
            os.unlink(temp_path)
            raise

    return temp_path


def write_pcm_s16le_wav(pcm_bytes: bytes, sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stream.wav") as output:
        temp_path = output.name

    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return temp_path


def remove_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
