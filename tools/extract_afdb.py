import argparse
import os
import tarfile
from glob import iglob

import pandas as pd
import psutil
import ray


n_cpu = psutil.cpu_count()
ray.init(num_cpus = n_cpu, ignore_reinit_error=True)

@ray.remote
def extract_tarfile(fn, gz_dir, file_number, split_number):
    tar = tarfile.open(fn)
    tar.extractall(os.path.join(gz_dir,split_number))
    tar.close()
    if file_number % 1000 ==0:
        print(file_number)
    return None

def extract_tarfiles(args):
    tar_file = os.path.join(args.tar_list_dir, f"{args.tar_file_name}.txt")
    with open(tar_file, 'r') as f:
        tar_list = f.readlines()
    tar_list = [tl.strip() for tl in tar_list]

    _ = ray.get([extract_tarfile.remote(fn, args.gz_dir, file_number, args.tar_file_name) for file_number, fn in enumerate(tar_list)])
    ray.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--tar_list_dir', type=str, default='datasets/afdb/tar_lists', help='a path for a file containing tar filenames')
    p.add_argument('--gz_dir', type=str, default='datasets/afdb/gz', help='a diretory for gz files')
    p.add_argument('--tar_file_name', type=str, default='0', help='a tar file name')

    args = p.parse_args()

    extract_tarfiles(args)