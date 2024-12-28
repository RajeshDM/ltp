import argparse
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=Path, help='Path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='Path to validation dataset')
    parser.add_argument('--size', default=32, type=int, help='Number of features per object')
    parser.add_argument('--iterations', default=30, type=int, help='Number of convolutions')
    parser.add_argument('--batch_size', default=16, type=int, help='Maximum size of batches')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate of training session')
    parser.add_argument('--l1', default=0.0, type=float, help='Strength of L1 regularization')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Strength of weight decay regularization')
    parser.add_argument('--gradient_accumulation', default=1, type=int, help='Number of gradients to accumulate before step')
    parser.add_argument('--max_samples_per_value', default=100, type=int, help='Maximum number of states per value per dataset')
    parser.add_argument('--patience', default=40, type=int, help='Patience for early stopping')
    parser.add_argument('--gradient_clip', default=0.1, type=float, help='Gradient clip value')
    parser.add_argument('--profiler', default=None, type=str, help='"simple", "advanced" or "pytorch"')
    parser.add_argument('--verbose', action='store_true', help='Print additional information during training')
    args = parser.parse_args()
    return args


def _load_datasets(args):
    print('Loading datasets...')
    from datasets.supervised.optimal import load_dataset, collate
    (train_dataset, predicates) = load_dataset(args.train, args.max_samples_per_value)
    (validation_dataset, _) = load_dataset(args.validation, args.max_samples_per_value)
    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_params)
    return predicates, train_loader, validation_loader


def _load_model(args, predicates):
    print('Loading model...')
    model_params = {
        "predicates": predicates,
        "hidden_size": args.size,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l1_factor": args.l1,
        "weight_decay": args.weight_decay,
    }
    from architecture.supervised.optimal import MaxModel as Model
    model = Model(**model_params)
    return model


def _load_trainer(args):
    print('Initializing trainer...')
    early_stopping = EarlyStopping(monitor='validation_loss', patience=args.patience)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor='validation_loss')
    trainer_params = {
        "num_sanity_val_steps": 0,
        "progress_bar_refresh_rate": 30 if args.verbose else 0,
        "callbacks": [early_stopping, checkpoint],
        "weights_summary": None,
        "auto_lr_find": True,
        "profiler": args.profiler,
        "accumulate_grad_batches": args.gradient_accumulation,
        "gradient_clip_val": args.gradient_clip,
    }
    if args.gpus > 0: trainer = pl.Trainer(gpus=args.gpus, auto_select_gpus=True, **trainer_params)
    else: trainer = pl.Trainer(**trainer_params)
    return trainer


if __name__ == "__main__":
    args = _parse_arguments()
    predicates, train_loader, validation_loader = _load_datasets(args)
    model = _load_model(args, predicates)
    trainer = _load_trainer(args)
    print('Training model...')
    trainer.fit(model, train_loader, validation_loader)
