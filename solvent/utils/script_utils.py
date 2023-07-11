# OpenFold (https://github.com/aqlaboratory/openfold)
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import re
import time

import numpy
import torch

from solvent.common import protein, residue_constants


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def prep_output(out, batch, config=None, subtract_plddt=False, multimer_ri_gap=200):
    plddt = out["plddt"]

    plddt_b_factors = numpy.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    template_domain_names = []
    template_chain_index = None

    if config is not None:
        no_recycling = config.MODEL.NUM_RECYCLE
    else:
        no_recycling = 3
    config_preset = 'finetuning'
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"config_preset={config_preset}",
    ])

    # For multi-chain FASTAs
    ri = batch["residue_index"]
    if 'chain_index' in batch:
        chain_index = batch['chain_index']
    else:
        chain_index = (ri - numpy.arange(ri.shape[0])) / multimer_ri_gap
        chain_index = chain_index.astype(numpy.int64)
        cur_chain = 0
        prev_chain_max = 0
        for i, c in enumerate(chain_index):
            if c != cur_chain:
                cur_chain = c
                prev_chain_max = i + cur_chain * multimer_ri_gap

            batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein

def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file