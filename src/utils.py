from pathlib import Path
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import noisereduce as nr
import librosa
import soundfile as sf
import subprocess


def extract_audio_from_video(video_file_path: str | Path, output_audio_file_path: str | Path):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio_file_path)

    audio_clip.close()
    video_clip.close()

def srt_timestamp_to_seconds(timestamp):
    return (
        timestamp.hours * 3600 +
        timestamp.minutes * 60 +
        timestamp.seconds +
        timestamp.milliseconds / 1000
    )

def to_array_fp32(audio: AudioSegment) -> np.array:
    return np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

def array_to_array_int16(audio: np.array) -> np.array:
    samples_int16_reconstructed = np.clip(audio, -1.0, 1.0)
    samples_int16_reconstructed = (samples_int16_reconstructed * 32768).astype(np.int16)
    return samples_int16_reconstructed

def adjust_noise_levels(redubbed_audio, audio_target: AudioSegment) -> AudioSegment:
    """
    Adjust noise level of redubbed audio to original audio noise level
    """
    noise_part = to_array_fp32(audio_target)  # Assume first 1s contains background noise
    redubbed_audio_noise = nr.reduce_noise(y=to_array_fp32(redubbed_audio), sr=sr, y_noise=noise_part)
    return AudioSegment(
        array_to_array_int16(redubbed_audio_noise).tobytes(),
        frame_rate=sr,
        sample_width=4,
        channels=1
    )


import warnings


def adjust_speed(audio_segment, speed_factor, min_factor=0.5, max_factor=2.0):
    """
    Adjust speed of synthesized audio segment to fit in original audio.

    Args:
        audio_segment (AudioSegment): Input audio segment.
        speed_factor (float): Speed factor in range from min_factor to max_factor to modify speed.
    """
    if speed_factor < min_factor or speed_factor > max_factor:
        warnings.warn(f"Speed factor {speed_factor} out of range.")

    # Export AudioSegment to a buffer
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)

    y, sr = librosa.load(buffer, sr=audio_segment.frame_rate)
    y_fast = librosa.effects.time_stretch(y, rate=speed_factor)
    y_fast = librosa.resample(y_fast, orig_sr=sr, target_sr=sr, res_type='kaiser_best')

    # Save sped-up audio back to a buffer
    output_buffer = BytesIO()
    sf.write(output_buffer, y_fast, sr, format="wav", subtype='PCM_24')
    output_buffer.seek(0)

    # Reload into AudioSegment
    return AudioSegment.from_file(output_buffer, format="wav")

def insert_segments_to_audio(wav_path, redubbed_audios, fade_level = 20):
    original_audio = AudioSegment.from_file(wav_path)

    for redubbed_segment in redubbed_audios:
        redubbed_audio = AudioSegment.from_file(redubbed_segment["fname"])
        start_time, end_time = redubbed_segment["start"], redubbed_segment["end"]

        required_duration = (end_time - start_time) * 1000
        if len(redubbed_audio) < required_duration:
            silence = AudioSegment.silent(duration=(required_duration - len(redubbed_audio)) // 2)
            redubbed_audio = silence + redubbed_audio + silence
            redubbed_audio = adjust_noise_levels(redubbed_audio, original_audio[start_time * 1000:end_time * 1000])
        else:
            speed_factor = len(redubbed_audio) / required_duration
            if speed_factor > 1.1:
                redubbed_audio = adjust_speed(redubbed_audio, speed_factor)

        volume_difference = original_audio.dBFS - redubbed_audio.dBFS
        redubbed_audio = redubbed_audio + volume_difference

        original_audio = original_audio[:start_time * 1000].fade_out(fade_level) + redubbed_audio.fade_in(
            fade_level).fade_out(fade_level) + original_audio[end_time * 1000:].fade_in(fade_level)
    return original_audio


def convert_and_merge_audio(video_path, audio_path, output_path):
    # Step 1: Convert Audio to AAC, Stereo
    converted_audio = audio_path.replace('.wav', '_converted.aac')
    subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-ar", "44100", "-ac", "2", "-c:a", "aac",
        converted_audio
    ], check=True)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-i", converted_audio,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-shortest",
        output_path
    ]

    # Step 2: Merge Audio with Video
    subprocess.run(command, check=True)

    print("Audio successfully added to the video.")