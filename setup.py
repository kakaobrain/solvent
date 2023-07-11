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
from setuptools import setup, Extension, find_packages
import subprocess
import ctypes

from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def get_nvidia_cc():
    """
    Returns a tuple containing the Compute Capability of the first GPU
    installed in the system (formatted as a tuple of strings) and an error
    message. When the former is provided, the latter is None, and vice versa.
    Adapted from script by Jan Schlüte t
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """
    CUDA_SUCCESS = 0

    libnames = [
        'libcuda.so', 
        'libcuda.dylib', 
        'cuda.dll',
        '/usr/local/cuda/compat/libcuda.so', # For Docker
    ]
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return None, "Could not load any of: " + ' '.join(libnames)

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        if error_str.value:
            return None, error_str.value.decode()
        else:
            return None, "Unknown error: cuInit returned %d" % result
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return None, error_str.value.decode()

    if nGpus.value < 1:
        return None, "No GPUs detected"

    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return None, error_str.value.decode()

    if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != CUDA_SUCCESS:
        return None, "Compute Capability not found"

    major = cc_major.value
    minor = cc_minor.value

    return (major, minor), None

version_dependent_macros = [
    '-DVERSION_GE_1_1',
    '-DVERSION_GE_1_3',
    '-DVERSION_GE_1_5',
]

extra_cuda_flags = [
    '-std=c++14',
    '-maxrregcount=50',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda'
]

def get_cuda_bare_metal_version(cuda_dir):
    if cuda_dir==None:
        print("CUDA is not found, cpu version is installed")
        return None, -1, 0
    else:
        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
        
        return raw_output, bare_metal_major, bare_metal_minor

compute_capabilities = set([
    (3, 7), # K80, e.g.
    (5, 2), # Titan X
    (6, 1), # GeForce 1000-series
])

compute_capabilities.add((7, 0))
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    compute_capabilities.add((8, 0))

compute_capability, _ = get_nvidia_cc()
if compute_capability is not None:
    compute_capabilities = set([compute_capability])

cc_flag = []
for major, minor in list(compute_capabilities):
    cc_flag.extend([
        '-gencode',
        f'arch=compute_{major}{minor},code=sm_{major}{minor}',
    ])

extra_cuda_flags += cc_flag

if bare_metal_major != -1:
    modules = [CUDAExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "solvent/utils/kernel/cuda_native/csrc/softmax_cuda.cpp",
            "solvent/utils/kernel/cuda_native/csrc/softmax_cuda_kernel.cu",
        ],
        include_dirs=[
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'solvent/utils/kernel/cuda_native/csrc/'
            )
        ],
        extra_compile_args={
            'cxx': ['-O3'] + version_dependent_macros,
            'nvcc': (
                ['-O3', '--use_fast_math'] +
                version_dependent_macros +
                extra_cuda_flags
            ),
        }
    )]
else:
    modules = [CppExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "solvent/utils/kernel/cuda_native/csrc/softmax_cuda.cpp",
            "solvent/utils/kernel/cuda_native/csrc/softmax_cuda_stub.cpp",
        ],
        extra_compile_args={
            'cxx': ['-O3'],
        }
    )]

setup(
    name='solvent',
    version='0.0.1',
    description='Protein Folding Framework',
    author='KakaoBrain',
    license='Apache License, Version 2.0',
    packages=find_packages(exclude=["tools"]),
    include_package_data=True,
    package_data={
        "solvent": [
            'utils/kernel/cuda_native/*',
            'utils/kernel/triton/*',
            'utils/kernel/cuda_native/csrc/*'
        ],
        "": ["resources/stereo_chemical_props.txt"]
    },
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7,'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)