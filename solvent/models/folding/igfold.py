# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from IgFold (https://github.com/Graylab/IgFold)
# Copyright 2022 The Johns Hopkins University
# https://github.com/Graylab/IgFold/blob/main/LICENSE.md

from detectron2.config import configurable

from igfold.model.components import IPATransformer
from igfold.utils.transforms import quaternion_to_matrix
from solvent.models.folding.build import FOLDING_REGISTRY

from .folding import FOLDING


@FOLDING_REGISTRY.register()
class IGFoldStructure(FOLDING):
    @configurable()
    def __init__(
        self,
        c_s,
        c_z,
        no_blocks,
        no_heads_ipa,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_blocks:
                Number of structure module blocks
        """
        super(IGFoldStructure, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.no_blocks = no_blocks
        self.no_heads_ipa = no_heads_ipa

        self.structure_ipa = IPATransformer(
            dim=c_s,
            pairwise_repr_dim=c_z,
            depth=no_blocks,
            heads=no_heads_ipa,
            require_pairwise_repr=True,
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "c_s": cfg.MODEL.TRUNK.SINGLE_DIM,
            "c_z": cfg.MODEL.TRUNK.PAIR_DIM,
            "no_blocks": cfg.MODEL.FOLDING.NUM_BLOCKS,
            "no_heads_ipa": cfg.MODEL.FOLDING.IPA.NUM_HEAD,
        }

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]
        z = evoformer_output_dict["pair"]
        
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        ipa_coords, ipa_translations, ipa_quaternions = self.structure_ipa(
            s,
            translations=None,
            quaternions=None,
            pairwise_repr=z,
            mask=mask.bool(),
        )
        
        ipa_rotations = quaternion_to_matrix(ipa_quaternions)
        outputs = {}
        outputs['single'] = s
        outputs['rot'] = ipa_rotations
        outputs['trans'] = ipa_translations
        outputs['final_atom_positions'] = ipa_coords

        return outputs