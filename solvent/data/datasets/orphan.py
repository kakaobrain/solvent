import glob
import json
import os
from datetime import datetime

from detectron2.data import DatasetCatalog, MetadataCatalog


def load_orphan(data_dir):
    cache_path = os.path.join(data_dir, 'chain_data_cache.json')
    with open(cache_path, 'r') as f:
        data = json.load(f)

    limit = datetime(2020, 5, 1)
    dataset_dicts = []
    for full_name, info in data.items():
        records = {}

        year, month, day = info['release_date'].split('-')
        rd = datetime(int(year), int(month), int(day))

        if rd < limit:
            continue

        file_id, chain_id = full_name.split('_')
        records['data_name'] = 'orphan'
        records['full_name'] = full_name
        records['file_id'] = file_id
        records['chain_id'] = chain_id
        records['file_name'] = os.path.join(data_dir, 'parsed_data', full_name+'.json')
        records['data_weight'] = 1.0

        dataset_dicts.append(records)

    return dataset_dicts

def register_orphan(name, data_dir):
    DatasetCatalog.register(name, lambda: load_orphan(data_dir))
    MetadataCatalog.get(name).set(
        evaluator_type="protein_folding",
    )

dataset_name = 'orphan'
dataset_dir = 'orphan'
register_orphan(dataset_name, os.path.join('datasets', dataset_dir))