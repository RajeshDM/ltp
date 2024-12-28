import argparse
import torch

from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from architecture import MaxModelBase as Model
from datasets.supervised.optimal import load_dataset, collate


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True, type=Path, help='Path to test directory')
    parser.add_argument('--model', required=True, type=Path, help='Path to model')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    args = parser.parse_args()
    return args


def _load_dataset(args):
    print('Loading datasets...')
    dataset, _ = load_dataset(args.test, 100000)
    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "num_workers": 4,
        "collate_fn": collate,
        "pin_memory": True,
        "shuffle": True,
    }
    loader = DataLoader(dataset, **loader_params)
    return loader


if __name__ == "__main__":
    args = _parse_arguments()

    print('Initializing model...')
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    model = Model.load_from_checkpoint(checkpoint_path=str(args.model), strict=False).to(device)
    dataset = _load_dataset()

    print('Begin testing...')
    model.eval()
    sum = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)

    with torch.no_grad():
        for input, label in dataset:
            for predicate, values in input[0].items():
                input[0][predicate] = values.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(input)
            error = torch.abs(output - label)
            sum = torch.add(sum, torch.sum(error))
            total = torch.add(total, label.nelement())

    print("{} ({} / {})".format(float(sum / total), int(sum), int(total)))
