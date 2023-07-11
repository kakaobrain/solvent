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

import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from solvent.common import protein, residue_constants
from solvent.models.heads.alphafold2 import lddt_ca
from solvent.utils.script_utils import prep_output
from solvent.utils.tensor_utils import tensor_tree_map
from solvent.utils.validation_metrics import drmsd, gdt_ha, gdt_ts


def _evaluate_predictions_on_antibody(gts, predictions):
    
    metrics = {
        'lddt_ca': 0,
        'drmsd_ca': 0,        
    }
    for batch, outputs in zip(gts, predictions):
        gt_coords = torch.tensor(batch["all_atom_positions"])[..., :5, :]
        pred_coords = torch.tensor(outputs["final_atom_positions"])[..., :5, :]
        all_atom_mask = torch.tensor(batch["all_atom_mask"])[..., :5]

        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-8,
            per_residue=False,
        )

        metrics["lddt_ca"] += lddt_ca_score.numpy()

        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )

        metrics["drmsd_ca"] += drmsd_ca_score.numpy()

    metrics["lddt_ca"] /= len(gts)
    metrics["drmsd_ca"] /= len(gts)
    return metrics

    
class AntibodyFoldingEvaluator(DatasetEvaluator):
    
    def __init__(
        self,
        dataset_name,
        tasks=('folding',),
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. 
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self.gt_keys = ['all_atom_positions', 'all_atom_mask']
        self.dt_keys = ['final_atom_positions']

    def reset(self):
        self._inputs = []
        self._predictions = []

    def process(self, inputs, outputs):
        gts = {}
        for k, v in inputs.items():
            if k in self.gt_keys:
                gts[k] = v[..., 0].to(self._cpu_device).tolist()

        dts = {}
        for k, v in outputs.items():
            if k in self.dt_keys:
                dts[k] = v.to(self._cpu_device).tolist()
        
        self._inputs.append(gts)
        self._predictions.append(dts)

    def evaluate(self):

        if self._distributed:
            comm.synchronize()
            inputs = comm.gather(self._inputs, dst=0)
            inputs = list(itertools.chain(*inputs))
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            inputs = self._inputs
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[ProteinFoldingEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._eval_predictions(inputs, predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, inputs, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        folding_results = predictions
        tasks = self._tasks

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "folding_results.json")
            # with PathManager.open(file_path, "wb") as f:
            #     torch.save(predictions, f)

        self._logger.info("Evaluating predictions")

        for task in sorted(tasks):
            assert task in {"folding"}, f"Got unknown task: {task}!"
            protein_eval = _evaluate_predictions_on_antibody(inputs, folding_results)
            res = self._derive_folding_results(protein_eval)
            self._results[task] = res

    def _derive_folding_results(self, protein_eval, task='folding'):
        results = protein_eval
        self._logger.info(
            "Evaluation results for {}: \n".format(task) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        return results