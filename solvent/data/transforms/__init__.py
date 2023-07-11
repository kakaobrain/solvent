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

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"
FEAT = {
    "aatype": [NUM_RES],
    "chain_index": [NUM_RES],
    "cdr_index": [NUM_RES],
    "all_atom_mask": [NUM_RES, None],
    "all_atom_positions": [NUM_RES, None, None],
    "alt_chi_angles": [NUM_RES, None],
    "atom14_alt_gt_exists": [NUM_RES, None],
    "atom14_alt_gt_positions": [NUM_RES, None, None],
    "atom14_atom_exists": [NUM_RES, None],
    "atom14_atom_is_ambiguous": [NUM_RES, None],
    "atom14_gt_exists": [NUM_RES, None],
    "atom14_gt_positions": [NUM_RES, None, None],
    "atom37_atom_exists": [NUM_RES, None],
    "backbone_rigid_mask": [NUM_RES],
    "backbone_rigid_tensor": [NUM_RES, None, None],
    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
    "chi_angles_sin_cos": [NUM_RES, None, None],
    "chi_mask": [NUM_RES, None],
    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
    "is_distillation": [],
    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
    "msa_row_mask": [NUM_MSA_SEQ],
    "no_recycling_iters": [],
    "pseudo_beta": [NUM_RES, None],
    "pseudo_beta_mask": [NUM_RES],
    "residue_index": [NUM_RES],
    "residx_atom14_to_atom37": [NUM_RES, None],
    "residx_atom37_to_atom14": [NUM_RES, None],
    "resolution": [],
    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
    "rigidgroups_group_exists": [NUM_RES, None],
    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
    "rigidgroups_gt_exists": [NUM_RES, None],
    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
    "seq_length": [],
    "seq_mask": [NUM_RES],
    "target_feat": [NUM_RES, None],
    "template_aatype": [NUM_TEMPLATES, NUM_RES],
    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
    "template_all_atom_positions": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_alt_torsion_angles_sin_cos": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
    "template_backbone_rigid_tensor": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_mask": [NUM_TEMPLATES],
    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
    "template_sum_probs": [NUM_TEMPLATES, None],
    "template_torsion_angles_mask": [
        NUM_TEMPLATES, NUM_RES, None,
    ],
    "template_torsion_angles_sin_cos": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "true_msa": [NUM_MSA_SEQ, NUM_RES],
    "use_clamped_fape": [],
}
    