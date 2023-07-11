# Detectron2 (https://github.com/facebookresearch/detectron2)
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
import logging
import pickle
from typing import Callable, Union

import numpy as np
import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)


class _TorchSerializedList(object):
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, lst: list):
        self._lst = lst

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(self._lst)
            )
        )
        self._lst = [_serialize(x) for x in self._lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)


_DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = _TorchSerializedList

class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
        self,
        lst: list,
        copy: bool = True,
        serialize: Union[bool, Callable] = True,
    ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool or callable): whether to serialize the stroage to other
                backend. If `True`, the default serialize method will be used, if given
                a callable, the callable will be used as serialize method.
        """
        self._lst = lst
        self._copy = copy
        if not isinstance(serialize, (bool, Callable)):
            raise TypeError(f"Unsupported type for argument `serailzie`: {serialize}")
        self._serialize = serialize is not False

        if self._serialize:
            serialize_method = (
                serialize
                if isinstance(serialize, Callable)
                else _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
            )
            logger.info(f"Serializing the dataset using: {serialize_method}")
            self._lst = serialize_method(self._lst)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy and not self._serialize:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]