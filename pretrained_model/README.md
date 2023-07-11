## Download pretrained language model
Download pretrained models at `pretrained_model`
| Shorthand | file name           | Download method|
|-----------|---------------------|-----------------|
| ESM-2(650M)| `esm2_t33_650M_UR50D.pt`                     | download models at [ESM official site](https://github.com/facebookresearch/esm#available-models)
|            | `esm2_t33_650M_UR50D-contact-regression.pt`
| ESM-2(150M)| `esm2_t30_150M_UR50D.pt`                     |
|            | `esm2_t30_150M_UR50D-contact-regression.pt`  |
| ESM-2(35M) | `esm2_t12_35M_UR50D.pt`                      |
|            | `esm2_t12_35M_UR50D-contact-regression.pt`   |
| OMEGAPLM   | `OmegaPLM.pt`                                | download models at [OmegaFold official site](https://github.com/HeliXonProtein/OmegaFold) and run `python tools/convert_solvent_weights.py` 
| Antiberty  | `igfold_1.ckpt`                              | download models at [Igfold official site](https://github.com/Graylab/IgFold) and use `IgFold/igfold_1.ckpt`