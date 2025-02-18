import os
from collections import defaultdict
from multiprocessing.managers import Value
from pathlib import Path
from pty import spawn

from elevenlabs import play
from elevenlabs.client import ElevenLabs
from pyannote.core.annotation import Annotation

from src.cloning_utils import extract_audio_for_speakers
from src.constants import ELEVENLABS_MODEL, ELEVENLABS_VOICE_SETTINGS

API_KEY = os.environ.get("ELEVENLABS_API_KEY")


class ElevenlabsSynthesizer:
    def __init__(self, diarization: Annotation, save_dir, api_key=API_KEY):
        self.client = ElevenLabs(
            api_key=api_key,
        )
        self.diarization = diarization
        self.create_speakers_dict()
        self.speakers_dst = Path(save_dir) / "saved_speaker_audios"

    def create_speakers_dict(self):
        self.speakers_segments = defaultdict(list)
        for segment, _, speaker in self.diarization.itertracks(yield_label=True):
            self.speakers_segments[speaker].append(segment)

    def create_voices(self, wav_path: str | Path, skip_api_call: bool):
        extract_audio_for_speakers(wav_path, self.diarization, self.speakers_dst)
        unique_speakers = list(self.speakers_segments.keys())
        if skip_api_call:
            return

        self.voices = {}

        if skip_api_call:
            return

        for spk in self.speakers_segments.keys():
            current_speaker_dst = self.speakers_dst / f"{spk}_audio.wav"

            voice = self.client.clone(
                name=spk,
                description=None,
                files=[current_speaker_dst],
            )
            self.voices[spk] = voice


    def clone(self, text, speaker, idx, audios_dir, skip_api_call) -> str | Path:
        fname = audios_dir / f"output_{idx}.wav"

        if skip_api_call:
            return fname

        if not self.voices.get(speaker):
           raise ValueError(f"Unknown speaker {speaker} passed for cloning.")

        print("==" * 10)
        print(f"Called 11 labs to synthesize {text}")
        audio = self.client.generate(text=text, voice=self.voices[speaker], voice_settings=ELEVENLABS_VOICE_SETTINGS,
                                     model=ELEVENLABS_MODEL)
        print("==" * 10)

        with open(fname, "wb") as audio_file:
            for chunk in audio:
                audio_file.write(chunk)
        return fname
