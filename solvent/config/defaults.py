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

from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "SSPF"

_C.MODEL.WEIGHTS = ""
_C.MODEL.NUM_RECYCLE = 3
_C.MODEL.BLOCKS_PER_CKPT = 1
_C.MODEL.CHUNK_SIZE = None
_C.MODEL.TUNE_CHUNK_SIZE = True
_C.MODEL.OFFLOAD_INFERENCE = False
_C.MODEL.USE_LMA = False
_C.MODEL.USE_FLASH = False
_C.MODEL._MASK_TRANS = False
_C.MODEL.SUPERVISED = True

# omegafold-specific
_C.MODEL.GATING = True # bool of gating method 

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# List of unsupervised features
_C.INPUT.UNSUPERVISED_FEATURES = [
    'aatype',
    'residue_index', 
    'msa', 
    'num_alignments', 
    'seq_length', 
    'between_segment_residues', 
    'deletion_matrix', 
    'no_recycling_iters',
    'chain_index',
    'cdr_index',
]
# List of supervised features
_C.INPUT.SUPERVISED_FEATURES = [
    'all_atom_mask', 
    'all_atom_positions', 
    'resolution', 
    'use_clamped_fape', 
    'is_distillation'
]
_C.INPUT.CROP = CN({"ENABLED": True})
_C.INPUT.CROP.SIZE = 256
_C.INPUT.CROP.MAX_TEMPLATE = 4
_C.INPUT.MSA_CLUSTER_FEATURES = False
_C.INPUT.USE_TEMPLATE = False
_C.INPUT.MAX_MSA_CLUSTERS = 1
_C.INPUT.MAX_EXTRA_MSA = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0

# ---------------------------------------------------------------------------- #
# Protein Language Model(PLM) options
# ---------------------------------------------------------------------------- #
_C.MODEL.EMBEDDER = CN()
_C.MODEL.EMBEDDER.NAME = 'esm2'
_C.MODEL.EMBEDDER.WEIGHT_KEY = 'esm2_t33_650M_UR50D'
_C.MODEL.EMBEDDER.FREEZE = True
_C.MODEL.EMBEDDER.NUM_LAYERS = 33
_C.MODEL.EMBEDDER.NUM_HEADS = 20
_C.MODEL.EMBEDDER.HIDDEN_DIM = 1280

# omegafold-specific
_C.MODEL.EMBEDDER.ALPHABET_SIZE = 21
_C.MODEL.EMBEDDER.MASKED_ALPHABET = 23 
_C.MODEL.EMBEDDER.PADDING_IDX = 21
_C.MODEL.EMBEDDER.PROJECT_DIM = 2560
_C.MODEL.EMBEDDER.ATTN_DIM = 256
_C.MODEL.EMBEDDER.NUM_RELPOS = 129
_C.MODEL.EMBEDDER.NUM_PSEUDO_MSA = 7
_C.MODEL.EMBEDDER.MASKED_RATIO = 0.12 # check this is neccessary 
_C.MODEL.EMBEDDER.SUBBATCH_SIZE = None
_C.MODEL.EMBEDDER.RELPOS_LEN = 32
# ---------------------------------------------------------------------------- #
# Trunk options
# ---------------------------------------------------------------------------- #
# base
_C.MODEL.TRUNK = CN()
_C.MODEL.TRUNK.NAME = 'EVOFORMER'
_C.MODEL.TRUNK.USE_MSA = False
_C.MODEL.TRUNK.NUM_BLOCKS = 48
_C.MODEL.TRUNK.MSA_DIM = 256
_C.MODEL.TRUNK.PAIR_DIM = 128
_C.MODEL.TRUNK.SINGLE_DIM = 384
_C.MODEL.TRUNK.MAX_RELATIVE_FEAT = 32
_C.MODEL.TRUNK.DROPOUT_RATE = 0.15
_C.MODEL.TRUNK.CLEAR_CACHE_BETWEEN_BLOCKS = False
# recycle embedder
_C.MODEL.TRUNK.RECYCLE_EMB = CN()
_C.MODEL.TRUNK.RECYCLE_EMB.NUM_BINS = 15
_C.MODEL.TRUNK.RECYCLE_EMB.MIN = 3.25
_C.MODEL.TRUNK.RECYCLE_EMB.MAX = 20.75

# struct embedder 
_C.MODEL.TRUNK.STRUCT_EMB = CN()
_C.MODEL.TRUNK.STRUCT_EMB.ENABLED = False
_C.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_MIN = 2.0
_C.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_MAX = 65
_C.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_NUM = 64
_C.MODEL.TRUNK.STRUCT_EMB.POS_BINS_MIN = -32
_C.MODEL.TRUNK.STRUCT_EMB.POS_BINS_MAX = 32
_C.MODEL.TRUNK.STRUCT_EMB.POS_BINS_NUM = 64
_C.MODEL.TRUNK.STRUCT_EMB.DIM = 16
_C.MODEL.TRUNK.RECYCLE_EMB.IGNORE_IDX = 0

# IGFold
_C.MODEL.TRUNK.TEMPLATE_IPA = CN()
_C.MODEL.TRUNK.TEMPLATE_IPA.NUM_BLOCKS = 2
_C.MODEL.TRUNK.TEMPLATE_IPA.NUM_HEADS = 8

# attention
_C.MODEL.TRUNK.ATTENTION = CN()
_C.MODEL.TRUNK.ATTENTION.DIM = 32
_C.MODEL.TRUNK.ATTENTION.NUM_HEADS = 8
# transition
_C.MODEL.TRUNK.TRANSITION = CN()
_C.MODEL.TRUNK.TRANSITION.INTERMEDIATE_FACTOR = 4
# outer product
_C.MODEL.TRUNK.OUTERPRODUCT = CN()
_C.MODEL.TRUNK.OUTERPRODUCT.DIM = 32
# triangle
_C.MODEL.TRUNK.TRIANGLE = CN()
_C.MODEL.TRUNK.TRIANGLE.DIM = 128
_C.MODEL.TRUNK.TRIANGLE.DROPOUT_RATE = 0.25
_C.MODEL.TRUNK.TRIANGLE.ATTN_DIM = 32
_C.MODEL.TRUNK.TRIANGLE.ATTN_NUM_HEADS = 4
_C.MODEL.TRUNK.TRIANGLE.COUNT = 2

# ---------------------------------------------------------------------------- #
# Folding options
# ---------------------------------------------------------------------------- #
# base
_C.MODEL.FOLDING = CN()
_C.MODEL.FOLDING.NAME = 'AlphafoldStructure'
_C.MODEL.FOLDING.SHARED = True
_C.MODEL.FOLDING.NUM_BLOCKS = 8
_C.MODEL.FOLDING.STOP_ROT_GRAD = True
# ipa
_C.MODEL.FOLDING.IPA = CN()
_C.MODEL.FOLDING.IPA.DIM = 16
_C.MODEL.FOLDING.IPA.DROPOUT_RATE = 0.1
_C.MODEL.FOLDING.IPA.NUM_HEAD = 12
_C.MODEL.FOLDING.IPA.NUM_SCALAR_QK = 16
_C.MODEL.FOLDING.IPA.NUM_POINT_QK = 4
_C.MODEL.FOLDING.IPA.NUM_SCALAR_V = 16
_C.MODEL.FOLDING.IPA.NUM_POINT_V = 8
# angle
_C.MODEL.FOLDING.ANGLE = CN()
_C.MODEL.FOLDING.ANGLE.NUM_BLOCKS = 2
_C.MODEL.FOLDING.ANGLE.DIM = 128
_C.MODEL.FOLDING.ANGLE.NUM_ANGLES = 7
# transition
_C.MODEL.FOLDING.TRANSITION = CN()
_C.MODEL.FOLDING.TRANSITION.NUM = 1
_C.MODEL.FOLDING.TRANSITION.SCALE_FACTOR = 10

# ---------------------------------------------------------------------------- #
# Head options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'AlphafoldHeads'
_C.MODEL.HEAD.EPS = 1e-8
# LDDT
_C.MODEL.HEAD.LDDT = CN()
_C.MODEL.HEAD.LDDT.NUM_BINS = 50
_C.MODEL.HEAD.LDDT.DIM = 128
_C.MODEL.HEAD.LDDT.LOSS_WEIGHT = 0.01
_C.MODEL.HEAD.LDDT.CUTOFF = 15.0
_C.MODEL.HEAD.LDDT.MIN_RES = 0.1
_C.MODEL.HEAD.LDDT.MAX_RES = 3.0
_C.MODEL.HEAD.LDDT.EPS = 1e-8
# Distogram
_C.MODEL.HEAD.DISTOGRAM = CN()
_C.MODEL.HEAD.DISTOGRAM.NUM_BINS = 64
_C.MODEL.HEAD.DISTOGRAM.LOSS_WEIGHT = 0.3
_C.MODEL.HEAD.DISTOGRAM.MIN_BIN = 2.3125
_C.MODEL.HEAD.DISTOGRAM.MAX_BIN = 21.6875
_C.MODEL.HEAD.DISTOGRAM.EPS = 1e-8
# Masked MSA
_C.MODEL.HEAD.MASKED_MSA = CN()
_C.MODEL.HEAD.MASKED_MSA.ENABLED = True
_C.MODEL.HEAD.MASKED_MSA.DIM = 23
_C.MODEL.HEAD.MASKED_MSA.LOSS_WEIGHT = 0.0 #2.0
_C.MODEL.HEAD.MASKED_MSA.PROFILE_PROB = 0.1
_C.MODEL.HEAD.MASKED_MSA.SAME_PROB = 0.1
_C.MODEL.HEAD.MASKED_MSA.UNIFORM_PROB = 0.1
_C.MODEL.HEAD.MASKED_MSA.REPLACE_FRACTION = 0.15
_C.MODEL.HEAD.MASKED_MSA.EPS = 1e-8
# Experimentally Resolved 
_C.MODEL.HEAD.EXP_RESOLVED = CN()
_C.MODEL.HEAD.EXP_RESOLVED.DIM = 37
_C.MODEL.HEAD.EXP_RESOLVED.LOSS_WEIGHT = 0.0 #0.01
_C.MODEL.HEAD.EXP_RESOLVED.MIN_RES = 0.1
_C.MODEL.HEAD.EXP_RESOLVED.MAX_RES = 3.0
_C.MODEL.HEAD.EXP_RESOLVED.EPS = 1e-8
# Chi
_C.MODEL.HEAD.CHI = CN()
_C.MODEL.HEAD.CHI.ANGLE_NORM_WEIGHT = 0.01
_C.MODEL.HEAD.CHI.CHI_WEIGHT = 0.5
_C.MODEL.HEAD.CHI.LOSS_WEIGHT = 1.0
_C.MODEL.HEAD.CHI.EPS = 1e-8
# violation
_C.MODEL.HEAD.VIOLATION = CN()
_C.MODEL.HEAD.VIOLATION.TOLERENCE_FACTOR = 12.0
_C.MODEL.HEAD.VIOLATION.CLASH_OVERLAP_TOLERANCE = 1.5
_C.MODEL.HEAD.VIOLATION.LOSS_WEIGHT = 1.0
_C.MODEL.HEAD.VIOLATION.EPS = 1e-8
# FAPE
_C.MODEL.HEAD.FAPE = CN()
_C.MODEL.HEAD.FAPE.LOSS_WEIGHT = 1.0
_C.MODEL.HEAD.FAPE.BACKBONE_CLAMP_DISTANCE = 10.0
_C.MODEL.HEAD.FAPE.BACKBONE_LOSS_UNIT_DISTANCE= 10.0
_C.MODEL.HEAD.FAPE.BACKBONE_LOSS_WEIGHT = 0.5
_C.MODEL.HEAD.FAPE.SIDECHAIN_CLAMP_DISTANCE = 10.0
_C.MODEL.HEAD.FAPE.SIDECHAIN_LENGTH_SCALE= 10.0
_C.MODEL.HEAD.FAPE.SIDECHAIN_LOSS_WEIGHT= 0.5
_C.MODEL.HEAD.FAPE.EPS = 1e-4
# TM
_C.MODEL.HEAD.TM = CN()
_C.MODEL.HEAD.TM.ENABLED = False
_C.MODEL.HEAD.TM.LOSS_WEIGHT = 0.1

# igfold-specific
_C.MODEL.HEAD.IGFOLD = CN()
_C.MODEL.HEAD.IGFOLD.IPA_NUM_BLOCKS = 2
_C.MODEL.HEAD.IGFOLD.IPA_HEAD = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Options: WarmupMultiStepLR, WarmupCosineLR.
# See detectron2/solver/build.py for definition.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.OPTIM = 'Adam'

_C.SOLVER.MAX_ITER = 800000         # 128 batch x 25000 = 3200000 / n batch

_C.SOLVER.BASE_LR = 0.001
# The end lr, only used by WarmupCosineLR
_C.SOLVER.BASE_LR_END = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
# Number of decays in WarmupStepWithFixedGammaLR schedule
_C.SOLVER.NUM_DECAYS = 3

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
# Whether to rescale the interval for the learning schedule after warmup
_C.SOLVER.RESCALE_INTERVAL = False

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of sequences per batch across all machines. 
_C.SOLVER.SEQ_PER_BATCH = 1
_C.SOLVER.SUBBATCH_SIZE = ''    # for omegafold

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})

_C.SOLVER.CLAMP_PROB = 0.9

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0

_C.TEST.AUG = CN({"ENABLED": False})

_C.TEST.PRECISE_BN = CN({"ENABLED": False})

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0



# deprecated
_C.MODEL.ALPHABET_SIZE = None
_C.MODEL.NODE_DIM = None
_C.MODEL.EDGE_DIM = None
_C.MODEL.TRUNK.RECYCLE_EMB.ROUGH_DIST_BINS = CN()
_C.MODEL.TRUNK.RECYCLE_EMB.ROUGH_DIST_BINS.MIN = None
_C.MODEL.TRUNK.RECYCLE_EMB.ROUGH_DIST_BINS.MAX = None
_C.MODEL.TRUNK.RECYCLE_EMB.ROUGH_DIST_BINS.BINS = None
_C.MODEL.TRUNK.RECYCLE_EMB.DIST_BINS = CN()
_C.MODEL.TRUNK.RECYCLE_EMB.DIST_BINS.MIN = None
_C.MODEL.TRUNK.RECYCLE_EMB.DIST_BINS.MAX = None
_C.MODEL.TRUNK.RECYCLE_EMB.DIST_BINS.BINS = None
_C.MODEL.TRUNK.RECYCLE_EMB.POS_BINS = CN()
_C.MODEL.TRUNK.RECYCLE_EMB.POS_BINS.MIN = None
_C.MODEL.TRUNK.RECYCLE_EMB.POS_BINS.MAX = None
_C.MODEL.TRUNK.RECYCLE_EMB.POS_BINS.BINS = None
