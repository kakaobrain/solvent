# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from ESM (https://github.com/facebookresearch/esm)
# Copyright (c) Meta Platforms, Inc. and affiliates.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from esm.pretrained import load_model_and_alphabet
from torch import nn

from solvent.common import residue_constants

from ..primitives import LayerNorm, Linear
from .build import EMBEDDER_REGISTRY
from .embedder import EMBEDDER


@EMBEDDER_REGISTRY.register()
class ESM(EMBEDDER):
    def __init__(self, cfg):
        """
        Args:
            name (str): name of language model
        """
        super().__init__()
        weight_key = cfg.MODEL.EMBEDDER.WEIGHT_KEY
        msa_dim = cfg.MODEL.TRUNK.MSA_DIM
        seq_dim = cfg.MODEL.EMBEDDER.HIDDEN_DIM
        plm_num_layer = cfg.MODEL.EMBEDDER.NUM_LAYERS
        num_head = cfg.MODEL.EMBEDDER.NUM_HEADS 
        pair_dim = cfg.MODEL.TRUNK.PAIR_DIM

        self.model, tokenizer = load_model_and_alphabet(weight_key)
        self.tokenizer = tokenizer.get_batch_converter()
        assert self.model.num_layers == plm_num_layer

        if cfg.MODEL.EMBEDDER.FREEZE:
            self.freeze()
            self.model.eval()
        else:
            self.freeze_head()

        self.esm_s_combine = nn.Parameter(torch.zeros(plm_num_layer + 1))
        self.embedding = nn.Embedding(residue_constants.restype_num + 3, msa_dim)
        self.esm_mlp = nn.Sequential(
            LayerNorm(seq_dim),
            Linear(seq_dim, msa_dim),
            nn.ReLU(),
            Linear(msa_dim, msa_dim),
        )
        in_dim = num_head * plm_num_layer
        self.esm_weights_mlp = nn.Sequential(
            LayerNorm(in_dim),
            Linear(in_dim, pair_dim),
            nn.ReLU(),
            Linear(pair_dim, pair_dim),
        )
    
    def forward(self, batched_inputs):
        aatype = batched_inputs['aatype']
        sequences = batched_inputs['seq']
        device = aatype.device

        # tokenize
        data = []
        for i, seq in enumerate(sequences):
            name = "batch_" + str(i)
            data.append((name, seq))
        batch_tokens = self.tokenizer(data)[-1]
        batch_tokens = batch_tokens.to(device)

        # forward PLM
        outputs = self.model(batch_tokens, repr_layers=range(self.model.num_layers+1), need_head_weights=True, return_contacts=False)
        
        # manipulate outputs
        representations = torch.stack(
            [v for _, v in sorted(outputs["representations"].items())], dim=2
        )
        representations = representations[:, 1:-1]
        
        attentions = outputs['attentions'][:, :, :, 1:-1, 1:-1]
        b, num_layers, num_heads, num_res, _ = attentions.shape
        attentions = attentions.view(b, num_layers * num_heads, num_res, num_res).permute(0, 2, 3, 1)
        
        # projection
        aa = aatype[..., 0]
        B = aa.shape[0]
        L = aa.shape[1]

        m = representations
        m = (self.esm_s_combine.softmax(0).unsqueeze(0) @ m).squeeze(2)
        m = self.esm_mlp(m)

        z = attentions
        z = self.esm_weights_mlp(z)

        m += self.embedding(aa)
        m = m.unsqueeze(1)

        batched_inputs['representations'] = m
        batched_inputs['attentions'] = z
        batched_inputs['masking_index'] = batched_inputs['seq_mask'].unsqueeze(1)
        
        return batched_inputs
