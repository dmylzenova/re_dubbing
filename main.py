import argparse
import os
from importlib.metadata import files
from pathlib import Path
import pysrt

from src.diarization import DiarizationPipeline
from src.utils import extract_audio_from_video, insert_segments_to_audio, convert_and_merge_audio, combine_sources
from src.process_subtitiles import align_subtitles_with_diarization
from src.elevenlabs_client import ElevenlabsSynthesizer
from src.cloning_utils import synthesize_edited_segments
from src.source_separation import separate_sources, save_separeted_audios

parser = argparse.ArgumentParser(
    description="Almost seamless Audio and Video Dubbing."
)

# Add required arguments
parser.add_argument(
    "-v", "--video_path",
    type=str,
    help="Path to the video file.",
    required=True,
)

parser.add_argument(
    "-o", "--original_transcription",
    type=str,
    help="Path to the original transcription file.",
    required = True,
)

parser.add_argument(
    "-e", "--edited_transcription",
    type=str,
    help="Path to the edited transcription file.",
    required=True,
)

parser.add_argument(
    "-b", "--background_music",
    action=argparse.BooleanOptionalAction,
    help="There is background music."
)


def main():
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    print("Video Path:", args.video_path)
    print("Original Transcription Path:", args.original_transcription)
    print("Edited Transcription Path:", args.edited_transcription)

    # Example functionality (replace with your actual processing logic)
    perform_dubbing(args.video_path, args.original_transcription, args.edited_transcription, args.background_music)


def perform_dubbing(video_path, original_transcription_file, edited_transcription_file, background_music=True):
    folder_path = Path(f"saved_files_{os.path.getsize(video_path)}")
    folder_path.mkdir(exist_ok=True)

    wav_path = folder_path / 'original_audio.wav'
    video_path = video_path
    
    # from pydub import AudioSegment
    # sound = AudioSegment.from_file(wav_path.with_suffix(".m4a"), format='m4a')
    # file_handle = sound.export(wav_path, format='wav')
    
    result_audio_path = folder_path / "result_audio.wav"
    result_video_path = folder_path / "result_video.mp4"

    # Load trascriptions
    original_transcription = pysrt.open(original_transcription_file)
    edited_transcription = pysrt.open(edited_transcription_file)

    # Extract audio from video
    extract_audio_from_video(video_path, wav_path)

    if background_music:
        # Run source separation
        wav_path, background_path = separate_sources(wav_path, folder_path)
        # wav_path, background_path = save_separeted_audios(separated_audios, folder_path)
    
    # Run diarization
    pipeline = DiarizationPipeline()
    diarization = pipeline.run_diarization(wav_path)

    # Align subtitiles with diarization
    alignment = align_subtitles_with_diarization(original_transcription, diarization)

    synthesizer = ElevenlabsSynthesizer(diarization, folder_path)
    synthesizer.create_voices(wav_path)

    redubbed_audios = synthesize_edited_segments(synthesizer, alignment, original_transcription, edited_transcription, folder_path)

    new_audio = insert_segments_to_audio(wav_path, redubbed_audios)

    new_audio.export(result_audio_path, format="wav")

    if background_music:
        background_path = combine_sources(wav_path, result_audio_path)
  
    convert_and_merge_audio(
        video_path=video_path,
        audio_path=result_audio_path.as_posix(),
        output_path=result_video_path.as_posix()
    )


if __name__ == "__main__":
    main()

