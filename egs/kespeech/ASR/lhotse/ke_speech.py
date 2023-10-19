"""
Description taken from the official website of wenetspeech
(https://wenet-e2e.github.io/WenetSpeech/)

We release a 10000+ hours multi-domain transcribed Mandarin Speech Corpus
collected from YouTube and Podcast. Optical character recognition (OCR) and
automatic speech recognition (ASR) techniques are adopted to label each YouTube
and Podcast recording, respectively. To improve the quality of the corpus,
we use a novel end-to-end label error detection method to further validate and
filter the data.

See https://github.com/wenet-e2e/WenetSpeech for more details about WenetSpeech
"""

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import argparse
# import librosa
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    compute_num_samples,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations

WETNET_SPEECH_PARTS = ("dev_phase1", "dev_phase2", "test", "train_phase1", "train_phase2")



def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="/home/zouyou/workspaces/ASR/newKaldi/icefall/egs/kespeech/ASR/KeSpeech/KeSpeech",
        help="Pathlike, the path of the data dir.",
    )

    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="all",
        help="Which parts of dataset to prepare, all for all the parts.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default='/home/zouyou/workspaces/ASR/newKaldi/icefall/egs/kespeech/ASR/data/manifests',
        help="Pathlike, the path where to write the manifests.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of workers to extract manifests.",
    )

    return parser


def prepare_ke_speech(args) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: Which parts of dataset to prepare, all for all the parts.
    :param output_dir: Pathlike, the path where to write the manifests.
    :num_jobs Number of workers to extract manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with
             the keys 'recordings' and 'supervisions'.
    """

    corpus_dir = Path(args.corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    subsets = WETNET_SPEECH_PARTS if "all" in args.dataset_parts else args.dataset_parts

    manifests = defaultdict(dict)
    for sub in subsets:
        if sub not in WETNET_SPEECH_PARTS:
            raise ValueError(f"No such part of dataset in WenetSpeech : {sub}")
        manifests[sub] = {"recordings": [], "supervisions": []}
        raw_manifests = read_file(sub, corpus_dir / 'Tasks' / 'ASR')

        with ProcessPoolExecutor(args.num_jobs) as ex:
            for recording, segments in tqdm(
                ex.map(
                    parse_utterance,
                    raw_manifests,
                    repeat(corpus_dir),
                    repeat(sub),
                ),
                desc="Processing KeSpeech manifests entries",
            ):
                manifests[sub]["recordings"].append(recording)
                manifests[sub]["supervisions"].append(segments)

        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(manifests[sub]["recordings"]),
            supervisions=SupervisionSet.from_segments(manifests[sub]["supervisions"]),
        )

        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )

        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"kespeech_supervisions_{sub}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"kespeech_recordings_{sub}.jsonl.gz")

        manifests[sub] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }

    return manifests


def parse_utterance(
    raw_manifests: Any, root_path: Path, sub: str
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:

    # 添加recording
    recording = Recording.from_file(
        path=root_path / raw_manifests["path"],
        recording_id=raw_manifests["id"],
    )    
    assert recording.sampling_rate == 16000, "The sampling rate is not 16000"
    # recording = Recording(
    #     id=raw_manifests["id"],
    #     sources=[
    #         AudioSource(
    #             type="file",
    #             channels=[0],
    #             source=str(root_path / raw_manifests["path"]),
    #         )
    #     ],
    #     sampling_rate=sampling_rate,
    #     num_samples=compute_num_samples(
    #         duration=duration, sampling_rate=sampling_rate
    #     ),
    #     duration=duration
    # )

    # 添加Supervision
    segments = SupervisionSegment(
        id=raw_manifests["id"],
        recording_id=raw_manifests["id"],
        start=0.0,
        duration=recording.duration,
        language="Chinese",
        text=raw_manifests["text"].strip(),
        speaker=raw_manifests['spkid']
    )

    return recording, segments


def read_file(sub: str, tag_path: Path):
    item_keys = ['text', 'utt2spk', 'utt2subdialect', 'wav.scp']
    
    item_dct = defaultdict(list)
    raw_manifests = []

    for item in item_keys:
        with open(tag_path / sub / item, 'r') as fr:
            item_dct[item.replace('.', '_')] = fr.readlines()
    
    for text, u2spk, u2subt, wav_scp in zip(item_dct['text'], item_dct['utt2spk'], 
                                             item_dct['utt2subdialect'], item_dct['wav_scp']):
        id1, txt = text.strip().split()
        id2, spk = u2spk.strip().split()
        id3, subt = u2subt.strip().split()
        id4, wav = wav_scp.strip().split()

        assert id1 == id2 == id3 == id4, \
            f"The utterance id is inconsistent, {id1}, {id2}, {id3}, {id4}"
        
        raw_manifests.append({'id': id1,
                              'text': txt,
                              'spkid': spk,
                              'subdialect': subt,
                              'path': wav})
    return raw_manifests


def main():
    parser = get_parser()
    args = parser.parse_args()

    prepare_ke_speech(args)


if __name__ == '__main__':
    main()