from src.utils import srt_timestamp_to_seconds


def align_subtitles_with_diarization(subtitles, diarization):
    diarization_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_list.append((turn.start, turn.end, speaker))

    subtitle_index = 0
    diarization_index = 0
    results = []

    while subtitle_index < len(subtitles) and diarization_index < len(diarization_list):
        subtitle = subtitles[subtitle_index]
        sub_start = srt_timestamp_to_seconds(subtitle.start)
        sub_end = srt_timestamp_to_seconds(subtitle.end)

        diarization_start, diarization_end, speaker = diarization_list[
            diarization_index
        ]

        if diarization_end <= sub_start:
            # Move to the next diarization segment
            diarization_index += 1
        elif diarization_start >= sub_end:
            # Move to the next subtitle segment
            subtitle_index += 1
        else:
            # Overlapping segment found
            overlap_start = max(sub_start, diarization_start)
            overlap_end = min(sub_end, diarization_end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                results.append(
                    {
                        "text": subtitle.text,
                        "speaker": speaker,
                        "start": sub_start,
                        "end": sub_end,
                    }
                )
                subtitle_index += 1
            else:
                diarization_index += 1

    return results
