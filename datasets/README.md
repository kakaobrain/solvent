# Data preparation

## [Protein Data Bank (PDBs) dataset](https://www.rcsb.org/)
Download dataset
```
bash datasets/download_pdb_mmcif.sh datasets
```
Preprocess dataset
```
python tools/preprocess_datasets.py \
    --pdb_dir datasets/pdb_mmcif/mmcif_files \                  # src cif files
    --output_dir datasets/pdb_mmcif/parsed_data                 # path to save parsed data
```
Expected dataset structure for PDBs
```
datasets/
    pdb_mmcif/
        mmcif_files/
        parsed_data/
        chain_data_cache.json
```

## [CAMEO Dataset](https://www.cameo3d.org/)
Download dataset
```
# 3 months data until 2022.06.25
python datasets/download_cameo.py \
    3-months 2022-06-25 datasets/cameo \
    --max_seqlen -1
```
Preprocess dataset
```
python tools/preprocess_datasets.py \
    --pdb_dir datasets/cameo/data_dir \             # src cif files
    --fasta_dir datasets/cameo/fasta_dir \          # src fasta files
    --output_dir datasets/cameo/parsed_data         # path to save parsed data
```
Expected dataset structure for CAMEO
```
datasets/
    cameo/
        data_dir/
        fasta_dir/
        parsed_data/
        chain_data_cache.json
```

## Alphafold predicted Uniref50 dataset
Download dataset
* Follow [official repository instructions](https://github.com/deepmind/alphafold/blob/main/afdb/README.md). 
* We use **v3** in all experiments
* Place downloaded files at `datasets/afdb/proteomes`

Preprocess dataset
`we introduce simple usage of the preparations. Please contact us for more details`
```
# extract *.tar file into *.gz files at `datasets/afdb/gz`
# we extract files separately with *.txt files in datasets/afdb/tar_lists
python tools/extract_afdb.py --tar_file_name 0

# generate afdb cache of all samples in `uniref_lists.txt` 
# (we generate cache just 3 sample as example)
python tools/generate_afdb_cache.py
```

Expected dataset structure for afdb
```
datasets/
    afdb/
        proteomes/
        gz/
        tar_lists/
        chain_data_cache.json
        uniref_lists.txt
```

## [SAbDab dataset](hhttps://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/)
Download dataset
* Download data and summary file in [SAbDab official site](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true#downloads)
* Locate downloaded file at datasets/SAbDab

Preprocess dataset
```
# truncate only Fv regions
python tools/extract_antibody_fv.py \
    --summary_file datasets/SAbDab/sabdab_summary_all.tsv \
    --db_path datasets/SAbDab/all_structures/chothia \
    --end_date 2021/03/31

# preprocess dataset
python tools/preprocess_multimer_datasets.py \
    --pdb_dir datasets/SAbDab/20210331/pdbs \
    --output_dir datasets/SAbDab/20210331/parsed_data
```

Expected dataset structure for SAbDab
```
datasets/
    SAbDab/
        all_structures/
            /raw
            /chothia
            /imgt
        20210331/
            /pdbs
            /parsed_data
            /chain_data_cache.json
        sabdab_summary_all.tsv
```
