# Installation

## Requirements
- Ubuntu18.04 with python 3.8
- CUDA 11.3
- Pytorch 1.12.1
- Scripts for data downloading require `rsync` and `aria2`

## Install dependencies and build solvent
-  prepare conda environment with installing all python dependencies (it takes a few minutes)
    ```
    conda env create --name=solvent -f environment.yml
    conda activate solvent
    ```
- install [xformers](https://github.com/facebookresearch/xformers) and [triton](https://github.com/openai/triton)
    ```
    pip install ninja
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@b31f4a1#egg=xformers
    pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
    ```
- installation via apt-get
    ```
    sudo apt-get update
    sudo apt-get install aria2 rsync
    ```
- Download folding resources. Downloaded at solvent/resources
    ```
    bash datasets/download_chemical_props.sh
    ```

- build solvent with compiling Openfold's CUDA kernel
    ```
    python setup.py install
    ```

## Install evaluation program
- Download and build [TM-score](https://zhanggroup.org/TM-score/TMscore.cpp) program and locate at TMscore/
    ```
    # expected program structure
    TMscore/
        TMscore
        TMscore.cpp
    ```