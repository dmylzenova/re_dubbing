import os
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation

from .constants import SPEAKER_DIARIZATION_MODEL

HUGGINGFACE_ACCESS_TOKEN = os.environ.get('HUGGINGFACE_ACCESS_TOKEN')

class DiarizationPipeline:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            SPEAKER_DIARIZATION_MODEL,
            use_auth_token=HUGGINGFACE_ACCESS_TOKEN)

        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def run_diarization(self, wav_path: str | Path, save: bool = True) -> Annotation:
        diarization = self.pipeline(wav_path)
        if save:
            with open(wav_path.parent / "diarization.rttm", "w") as f:
                diarization.write_rttm(f)
        return diarization
