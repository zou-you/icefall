"""
About the Aishell corpus
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.
publicly available on https://www.openslr.org/33
"""

import argparse
import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus-dir",
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
    )

    return parser


def text_normalize(line: str):
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/aishell_data_prep.sh#L54
    sed 's/ａ/a/g' | sed 's/ｂ/b/g' |\
    sed 's/ｃ/c/g' | sed 's/ｋ/k/g' |\
    sed 's/ｔ/t/g' > $dir/transcripts.t

    """
    line = line.replace("ａ", "a")
    line = line.replace("ｂ", "b")
    line = line.replace("ｃ", "c")
    line = line.replace("ｋ", "k")
    line = line.replace("ｔ", "t")
    line = line.upper()
    return line


def prepare_self_data(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    transcript_ASR240_path = corpus_dir / "ASR240/transcript/text"
    transcript_TTS279_path = corpus_dir / "TTS279/transcript/text"
    transcript_dict = {}
    with open(transcript_ASR240_path, "r", encoding="utf-8") as f_asr240, \
        open(transcript_TTS279_path, "r", encoding="utf-8") as f_tts279:
        for f in [f_asr240, f_tts279]:
            for line in f.readlines():
                idx_transcript = line.split()
                content = " ".join(idx_transcript[1:])
                content = text_normalize(content)
                transcript_dict[idx_transcript[0]] = content

    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev"]
    for part in tqdm(
        dataset_parts,
        desc="Process self_data audio.",
    ):
        logging.info(f"Processing self_data subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, text)
        recordings = []
        supervisions = []
        wav_ASR240_path = corpus_dir / "ASR240" / "wav" / f"{part}"
        wav_TTS279_path = corpus_dir / "TTS279" / "wav" / f"{part}"
        for wav_path in [wav_ASR240_path, wav_TTS279_path]:
            for audio_path in wav_path.rglob("**/*.wav"):
                idx = audio_path.stem
                if idx not in transcript_dict:
                    logging.warning(f"No transcript: {idx}")
                    logging.warning(f"{audio_path} has no transcript.")
                    continue
                text = transcript_dict[idx]
                if not audio_path.is_file():
                    logging.warning(f"No such file: {audio_path}")
                    continue
                recording = Recording.from_file(audio_path)
                recordings.append(recording)
                segment = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    text=text.strip(),
                )
                supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"self_data_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"self_data_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    prepare_self_data(corpus_dir=args.corpus_dir,
                      output_dir=args.output_dir)