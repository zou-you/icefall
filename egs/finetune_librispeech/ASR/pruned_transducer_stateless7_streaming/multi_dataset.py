# Copyright      2023  Xiaomi Corp.        (authors: Zengrui Jin)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import logging
import re
from pathlib import Path
from typing import Dict, List

import lhotse
from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, fbank_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - aishell_cuts_train.jsonl.gz
            - self_data_cuts_train.jsonl.gz
        """
        self.fbank_dir = Path(fbank_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # AISHELL-1
        logging.info("Loading Aishell-1 in lazy mode")
        aishell_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_train.jsonl.gz"
        )

        # SELF-DATA
        logging.info("Loading Self-Data in lazy mode")
        self_data_cuts = load_manifest_lazy(
            self.fbank_dir / "self_data_cuts_train.jsonl.gz")

        return CutSet.mux(
            aishell_cuts,
            self_data_cuts,
            weights=[
                len(aishell_cuts),
                len(self_data_cuts),
            ],
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # AISHELL
        logging.info("Loading Aishell DEV set in lazy mode")
        aishell_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_dev.jsonl.gz"
        )

        return aishell_dev_cuts
        # SELF-DATA
        # logging.info("Loading Self-Data DEV set in lazy mode")
        # self_data_dev_cuts = load_manifest_lazy(
        #     self.fbank_dir / "self_data_cuts_dev.jsonl.gz"
        # )

        # return CutSet.mux(
        #     aishell_dev_cuts,
        #     self_data_dev_cuts,
        #     weights=[
        #         len(aishell_dev_cuts),
        #         len(self_data_dev_cuts),
        #     ],
        # )

    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        # AISHELL
        logging.info("Loading Aishell set in lazy mode")
        aishell_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_test.jsonl.gz"
        )
        aishell_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_dev.jsonl.gz"
        )

        # SELF-DATA
        logging.info("Loading Self-Data DEV set in lazy mode")
        self_data_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "self_data_cuts_dev.jsonl.gz"
        )

        return CutSet.mux(
            aishell_test_cuts,
            aishell_dev_cuts,
            self_data_dev_cuts,
            weights=[
                len(aishell_test_cuts),
                len(aishell_dev_cuts),
                len(self_data_dev_cuts),
            ],
        )
