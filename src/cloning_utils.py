from pathlib import Path

from pyannote.core import Annotation, Segment
from pydub import AudioSegment
from collections import defaultdict

from src.utils import srt_timestamp_to_seconds
from constants import MIN_TEXT_LEN

def extract_consecutive_segments(diarization):
    """
    Extract list of consecutive segments for each speaker.
    """
    speaker_segments = defaultdict(list)
    current_label = None
    current_sequence = []
    print('8'* 88)
    for segment, _, label in diarization.itertracks(yield_label=True):
        current_label = label if current_label is None else current_label
        print('label', label, 'current_label', current_label)
        if label == current_label:
            current_sequence.append(segment)
        else:
            speaker_segments[current_label].append(current_sequence)

            current_label = label
            current_sequence = [segment]

    if current_label not in speaker_segments:
        speaker_segments[current_label].append(current_sequence)
    print('speaker_segments', len(speaker_segments))
    return speaker_segments


# # Step 2: Merge Adjacent Segments for Each Speaker
def merge_segments(list_of_segments: list, min_gap=1.0):
    """
    Merge consequtive segments with natural pauses on condition they are close enough.
    """
    result = []
    for segments in list_of_segments:
        merged = [segments[0]]

        for current in segments[1:]:
            previous = merged[-1]
            if current.start - previous.end <= min_gap:
                merged[-1] = Segment(previous.start, max(previous.end, current.end))
            else:
                merged.append(current)
        result.extend(merged)

    return result


def combine_audio_segments(audio: AudioSegment, segments: list, target_duration_ms: int = 20_000,
                    ) -> AudioSegment:
    """
    Combine audio segments to one without careful stitching as it is not that important for cloning.
    """
    combined_audio = AudioSegment.empty()
    current_duration = 0

    for segment in segments:

        segment_audio = audio[int(segment.start * 1000):int(segment.end * 1000)]
        segment_duration = len(segment_audio)

        remaining_duration = target_duration_ms - current_duration

        if remaining_duration <= 0:
            break

        if segment_duration <= remaining_duration:
            combined_audio += segment_audio
            current_duration += segment_duration
        else:
            combined_audio += segment_audio[:remaining_duration]
            current_duration += remaining_duration
            break
    print('target_duration_ms', target_duration_ms)
    return combined_audio[:target_duration_ms]


def extract_audio_for_speakers(audio_file: Path, diarization: Annotation, speakers_dst, target_duration=20):
    """
    Extract audio for cloning each speaker based on non-overlapping merged segments.
    """
    audio = AudioSegment.from_file(audio_file)
    speaker_segments = extract_consecutive_segments(diarization)
    target_duration = target_duration * 1000
    for speaker, segments in speaker_segments.items():
        print('speaker', speaker)
        merged_segments = merge_segments(segments)

        if not merged_segments:
            print('notttttttt')
            continue  # Skip if no valid segments

        merged_segments = sorted(merged_segments, key=lambda s: s.duration, reverse=True)

        speaker_audio = combine_audio_segments(audio, merged_segments, target_duration)
        print('len', len(speaker_audio))
        speakers_dst.mkdir(exist_ok=True)
        if speaker_audio.duration_seconds > 0:
            speaker_audio.export(f"{speakers_dst}/{speaker}_audio.wav", format="wav")
            print(f"Saved audio for speaker {speaker}")


def synthesize_edited_segments(synthesizer, alignment, original_transcription, edited_transcription, folder_path, enlarge_segments=True):
    audios_dir = folder_path / "redubbed_audios"
    audios_dir.mkdir(exist_ok=True)
    redubbed_audios = []
    edited_indices = set()
    
    for idx, (sub1, sub2) in enumerate(zip(original_transcription, edited_transcription)):
        if sub1.text != sub2.text and idx not in edited_indices:
            print(f"Difference at index {sub1.index}:")
            print(f"File1: {sub1.text}")
            print(f"File2: {sub2.text}")
            print(f"Time {sub1.start, sub1.end}, {sub2.start, sub2.end}")
            
            aligned_data = alignment[idx]
            print('aligned_data', aligned_data)

            speaker = aligned_data["speaker"]
            start_time = srt_timestamp_to_seconds(sub1.start)
            end_time = srt_timestamp_to_seconds(sub1.end)
            text = sub2.text

            # Expand segment if text too short and there is a possibility
            if enlarge_segments:
                if len(sub2.text) < MIN_TEXT_LEN:
                    if idx > 0 and alignment[idx - 1]["speaker"] == speaker:
                        start_time = srt_timestamp_to_seconds(original_transcription[idx - 1].start)
                        text = original_transcription[idx - 1].text + " " + text
                    if idx < len(original_transcription) - 1 and alignment[idx + 1]["speaker"] == speaker:
                        end_time = srt_timestamp_to_seconds(original_transcription[idx + 1].end)
                        text = text + " " + original_transcription[idx + 1].text
                        edited_indices.add(idx + 1)

            audio_path = synthesizer.clone(text, speaker, idx, audios_dir)

            segment_data = {"fname": audio_path,
                            "start": start_time,
                            "end": end_time,
                            }
            redubbed_audios.append(segment_data)
    print("Finished synthesizing audios.")
    return redubbed_audios
