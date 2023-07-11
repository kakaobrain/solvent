# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import multiprocessing as mp
import os
import string
import time

import tqdm
from detectron2.utils.logger import setup_logger
from predictor import FoldingDemo

from solvent.config import get_cfg
from solvent.data.parsers import parse_fasta


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Unifolds demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/sspf/base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger(name="solvent")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = FoldingDemo(cfg)

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(sorted(args.input), disable=not args.output):
        start_time = time.time()
        with open(path, 'r') as f:
            data = f.read()
        seqs, tags = parse_fasta(data)

        if len(seqs) == 1:
            sequence = seqs[0]
            file_name = tags[0]
            chain_index = None
        elif len(seqs) == 2:
            sequence = seqs[0] + seqs[1]
            file_name = os.path.splitext(os.path.basename(path))[0] + '_HL'
            
            chain_tag = string.ascii_uppercase
            chain_index = [chain_tag.index('H')] * len(seqs[0]) + [chain_tag.index('L')] * len(seqs[1])
        
        demo.run_on_sequence(sequence, chain_index, output_dir=output_dir, file_name=file_name, skip_relaxation=True)