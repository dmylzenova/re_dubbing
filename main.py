import argparse
import os
from pathlib import Path

import pysrt

from src.cloning_utils import synthesize_edited_segments

from src.diarization import DiarizationPipeline
from src.elevenlabs_client import ElevenlabsSynthesizer
from src.process_subtitiles import align_subtitles_with_diarization
from src.source_separation import separate_sources
from src.utils import (
    combine_sources,
    convert_and_merge_audio,
    extract_audio_from_video,
    insert_segments_to_audio,
)

parser = argparse.ArgumentParser(description="Almost seamless Audio and Video Dubbing.")

# Add required arguments
parser.add_argument(
    "-v",
    "--video_path",
    type=str,
    help="Path to the video file.",
    required=True,
)

parser.add_argument(
    "-o",
    "--original_transcription",
    type=str,
    help="Path to the original transcription file.",
    required=True,
)

parser.add_argument(
    "-e",
    "--edited_transcription",
    type=str,
    help="Path to the edited transcription file.",
    required=True,
)

parser.add_argument(
    "-b",
    "--background_music",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Call source separation because there is background music."
)

parser.add_argument(
    "-s", "--skip_api_call",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Call clonning API"
)



def main():
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    print("Video Path:", args.video_path)
    print("Original Transcription Path:", args.original_transcription)
    print("Edited Transcription Path:", args.edited_transcription)

    perform_dubbing(args.video_path, args.original_transcription, args.edited_transcription, args.background_music, args.skip_api_call)


def perform_dubbing(video_path, original_transcription_file, edited_transcription_file, background_music=True, skip_api_call=True):
    folder_path = Path(f"saved_files_{os.path.getsize(video_path)}")
    folder_path.mkdir(exist_ok=True)
    print('skip_api_call', skip_api_call)
    wav_path = folder_path / 'original_audio.wav'

    result_audio_path = folder_path / "result_audio.wav"
    result_video_path = Path(video_path).parent / "result_video.mp4"

    # Load trascriptions
    original_transcription = pysrt.open(original_transcription_file)
    edited_transcription = pysrt.open(edited_transcription_file)

    # Extract audio from video
    extract_audio_from_video(video_path, wav_path)

    if background_music:
        # Run source separation
        wav_path, background_path = separate_sources(wav_path, folder_path)

    # Run diarization
    pipeline = DiarizationPipeline()
    diarization = pipeline.run_diarization(wav_path)

    # Align subtitiles with diarization
    alignment = align_subtitles_with_diarization(original_transcription, diarization)

    synthesizer = ElevenlabsSynthesizer(diarization, folder_path)
    synthesizer.create_voices(wav_path, skip_api_call)

    redubbed_audios = synthesize_edited_segments(
        synthesizer,
        alignment,
        original_transcription,
        edited_transcription,
        folder_path,
        skip_api_call=skip_api_call
    )

    new_audio = insert_segments_to_audio(wav_path, redubbed_audios)

    new_audio.export(result_audio_path, format="wav")

    if background_music:
        result_audio_path = combine_sources(result_audio_path, background_path)

    convert_and_merge_audio(
        video_path=video_path,
        audio_path=result_audio_path.as_posix(),
        output_path=result_video_path.as_posix(),
    )


if __name__ == "__main__":
    main()
