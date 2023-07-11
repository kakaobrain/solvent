# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from IgFold (https://github.com/Graylab/IgFold)
# Copyright 2022 The Johns Hopkins University
# https://github.com/Graylab/IgFold/blob/main/LICENSE.md

import torch
import transformers
from torch import nn

from ..primitives import LayerNorm, Linear
from .build import EMBEDDER_REGISTRY
from .embedder import EMBEDDER


@EMBEDDER_REGISTRY.register()
class Antiberty(EMBEDDER):
    def __init__(self, cfg):
        """
        Args:
            name (str): name of language model
        """
        super().__init__()
        weight_key = cfg.MODEL.EMBEDDER.WEIGHT_KEY
        node_dim = cfg.MODEL.TRUNK.MSA_DIM
        edge_dim = cfg.MODEL.TRUNK.PAIR_DIM
        seq_dim = cfg.MODEL.EMBEDDER.HIDDEN_DIM
        num_head = cfg.MODEL.EMBEDDER.NUM_HEADS
        plm_num_layer = cfg.MODEL.EMBEDDER.NUM_LAYERS

        weights = torch.load(weight_key)
        state_dict = {}
        for k, v in weights['state_dict'].items():
            if "bert_model" in k:
                k = k.replace("bert_model.", "")
                state_dict[k] = v
        config = weights['hyper_parameters']['config']

        self.tokenizer = config["tokenizer"]
        self.vocab_size = len(self.tokenizer.vocab)
        self.model = transformers.BertModel(config["bert_config"])
        self.model.load_state_dict(state_dict, strict=True)

        if cfg.MODEL.EMBEDDER.FREEZE:
            self.freeze()
            self.model.eval()

        self.node_linear = nn.Sequential(
            LayerNorm(seq_dim),
            Linear(seq_dim, node_dim),
            nn.ReLU(),
            Linear(node_dim, node_dim),
        )
        in_dim = num_head * plm_num_layer
        self.edge_linear = nn.Sequential(
            LayerNorm(in_dim),
            Linear(in_dim, edge_dim),
            nn.ReLU(),
            Linear(edge_dim, edge_dim),
        )

    
    def tokenize(self, sequences, device):
        modified_seqs = []
        for seq in sequences:
            seq = seq.replace('<', '[')
            seq = seq.replace('>', ']')
            seq = seq.replace('pad', 'PAD')
            modified_seqs.append(seq)
        tokens = self.tokenizer.batch_encode_plus(modified_seqs, return_tensors="pt")['input_ids']
        return tokens.to(device)

    def forward(self, batched_inputs):
        '''
        output:
            representation.shape = (B, msa_dim, SEQ_LEN, DIM)
            attentions.shape = (B, SEQ_LEN, SEQ_LEN, DIM)
        '''
        device = batched_inputs['aatype'].device
        sequences = batched_inputs['seq']
        
        tokens = self.tokenize(sequences, device)
        output = self.model(tokens, output_hidden_states=True, output_attentions=True)

        feats = output.hidden_states[-1]
        feats = feats[:, 1:-1]

        attn = torch.cat(
            output.attentions,
            dim=1,
        )
        attn = attn[:, :, 1:-1, 1:-1]
        attn = attn.permute(0, 2, 3, 1)

        feats = self.node_linear(feats)
        attn = self.edge_linear(attn)


        batched_inputs['representations'] = feats.unsqueeze(1)
        batched_inputs['attentions'] = attn
        batched_inputs['masking_index'] = batched_inputs['seq_mask'].unsqueeze(1)
        
        return batched_inputs