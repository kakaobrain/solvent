import argparse
import torch


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument('--src-model-path', type=str, default='model2.pt', help='dataset name')
    p.add_argument('--new-model-path', type=str, default='pretrained_model/OmegaPLM.pt', help='dataset name')
    args = p.parse_args()

    src_model = torch.load(args.src_model_path)

    new_model = {}
    for k, v in src_model.items():
        if 'omega_plm' not in k:
            continue
        new_k = k.replace('omega_plm.', '')
        new_model[new_k] = v


    torch.save(new_model, args.new_model_path)