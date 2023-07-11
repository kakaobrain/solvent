# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from OpenFold (https://github.com/aqlaboratory/openfold)
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

import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..common import protein, residue_constants
from . import mmcif_parsing, parsers

FeatureDict = Mapping[str, np.ndarray]


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features

def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features

def process_msa_feats(
        input_sequence: Optional[str] = None,
    ) -> Mapping[str, Any]:

        def _get_msas(sequence):
            msa_data = {}
            msa_data["dummy"] = {
                "msa": [sequence],
                "deletion_matrix": [[0 for _ in sequence]],
            }
            msas, deletion_matrices = zip(*[
                (v["msa"], v["deletion_matrix"]) for v in msa_data.values()
            ])
            return msas, deletion_matrices

        msas, deletion_matrices = _get_msas(input_sequence)
        msa_features = make_msa_features(msas=msas, deletion_matrices=deletion_matrices)

        return msa_features

def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats

def empty_template_feats(n_res) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions": 
            np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }

def process_mmcif(

    mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
    alignment_dir: str = None,
    chain_id: Optional[str] = None,
    alignment_index: Optional[str] = None,
) -> FeatureDict:
    """
        Assembles features for a specific chain in an mmCIF object.

        If chain_id is None, it is assumed that there is only one chain
        in the object. Otherwise, a ValueError is thrown.
    """
    if chain_id is None:
        chains = mmcif.structure.get_chains()
        chain = next(chains, None)
        if chain is None:
            raise ValueError("No chains in mmCIF file")
        chain_id = chain.id

    mmcif_feats = make_mmcif_features(mmcif, chain_id)

    input_sequence = mmcif.chain_to_seqres[chain_id]    
    msa_features = process_msa_feats(input_sequence)

    return {**mmcif_feats, **msa_features}

def make_protein_features(
    protein_object: protein.Protein, 
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask
    chain_index = protein_object.chain_index

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)
    
    pdb_feats["chain_index"] = chain_index

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats

def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if(is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats

def process_pdb(
        pdb_path: str,
        is_distillation: bool = True,
        chain_id: Optional[str] = None,
        _structure_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype) 
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object, 
            description, 
            is_distillation=is_distillation
        )

        msa_features = process_msa_feats(input_sequence)

        return {**pdb_feats, **msa_features}