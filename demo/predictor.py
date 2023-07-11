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
import os
import string

import numpy as np
import torch

from solvent.common import protein
from solvent.engine import DefaultPredictor
from solvent.utils.script_utils import prep_output
from solvent.utils.tensor_utils import tensor_tree_map


class FoldingDemo(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def pad_inputs(self, sequence):
        num_res = len(sequence)
        padded_size = (int(num_res / 8) + 1) * 8
        num_pad = padded_size - num_res
        sequence += ''.join(['X'] * num_pad)
        return sequence, num_pad
    
    def unpad_features(self, inputs, outputs, num_pad):
        inputs = {
            'aatype': inputs['aatype'][:-num_pad],
            'residue_index': inputs['residue_index'][:-num_pad],
        }
        outputs = {
            'final_atom_positions': outputs['final_atom_positions'][:-num_pad],
            'final_atom_mask': outputs['final_atom_mask'][:-num_pad],
            'plddt': outputs['plddt'][:-num_pad],
        }
        return inputs, outputs


    def run_on_sequence(self, sequence, chain_index=None, output_dir=None, file_name=None, skip_relaxation=False, start_num=0):
        """
        Args:
            sequence

        Returns:
            predictions (dict): the output of the model.
        """
        
        # pad seq length as muliple of 8
        sequence, num_pad = self.pad_inputs(sequence)

        predictions, processed_feature_dict = self.predictor(sequence, chain_index)
        
        processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].squeeze(0).cpu()), 
                processed_feature_dict
            )
        predictions = tensor_tree_map(lambda x: np.array(x.squeeze(0).cpu()), predictions)

        # return back to original size
        processed_feature_dict, predictions = self.unpad_features(
            processed_feature_dict, 
            predictions,
            num_pad
        )

        unrelaxed_protein = prep_output(
            predictions, 
            processed_feature_dict,
            self.cfg,
        )

        unrelaxed_output_path = os.path.join(
            output_dir, f'{file_name}_unrelaxed.pdb'
        )

        chain_id = file_name.split('_')[1]
        if len(chain_id) == 1:
            chain_tag = string.ascii_uppercase
            residue_index = chain_tag.index(chain_id)
            unrelaxed_protein.chain_index.setfield(residue_index, dtype=np.int)

        with open(unrelaxed_output_path, 'w') as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))