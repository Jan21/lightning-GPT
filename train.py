from argparse import ArgumentParser
from urllib.request import urlopen

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from lightning_gpt import callbacks, data, gpt_models, rnn_models



def get_gpt(train_dataset,args):
    extra_kwargs = {}
    GPT_class = gpt_models.NanoGPT
    extra_kwargs["dropout"] = 0.1

    if args.strategy == "deepspeed":
        if GPT_class == gpt_models.MinGPT:
            GPT_class = gpt_models.DeepSpeedMinGPT
        elif GPT_class == gpt_models.NanoGPT:
            GPT_class = gpt_models.DeepSpeedNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with DeepSpeed")
        extra_kwargs["offload"] = False

    model = GPT_class(
        vocab_size=train_dataset.vocab_size,
        #vocab_size=50257,
        block_size=train_dataset.block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        **extra_kwargs,
    )
    return model


def get_RNN(train_dataset,args):
    args.model_type = 'LSTM'
    args.n_embd = 1024
    args.n_hidden = 1024
    args.n_layer = 2
    args.dropout = 0
    args.tied = True
    
    model = rnn_models.LSTM(
        args.model_type, 
        train_dataset.vocab_size,
        args.n_embd, 
        args.n_hidden, 
        args.n_layer,
        args.batch_size, 
        args.learning_rate,
        args.wdecay,
        args.optimizer,
        args.dropout, 
        args.tied)
    return model


def main(args):
    train_file = f'{args.data_folder}/train.bin'
    test_file = f'{args.data_folder}/val.bin'
    tokenizer_file = f'{args.data_folder}/tokenizer{args.tokenizer_type}.json'

    train_dataset = data.cc_czech_Dataset(train_file, args.block_size, tokenizer_file)
    test_dataset = data.cc_czech_Dataset(test_file, args.block_size, tokenizer_file)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)

    if args.model_type[:3] == "rnn":
        model = get_RNN(train_dataset=train_dataset,args=args)
    elif args.model_type[:3] == "gpt":
        model = get_gpt(train_dataset=train_dataset,args=args)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."
                "Please install torch >= 1.14 or disable compile."
            )
        model = torch.compile(model)

    callback_list = []

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.CUDAMetricsCallback())

    wandb_logger = WandbLogger(project=args.project_name, name=args.data_folder, save_dir=args.data_folder)
    trainer = L.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=callback_list,
        accelerator="auto",
        logger=wandb_logger,
        devices=args.devices,
        strategy='ddp',
        precision=16,
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    L.seed_everything(42)

    parser = ArgumentParser()
    parser = L.Trainer.add_argparse_args(parser)
    parser.add_argument("--project_name", default='cc', type=str)
    parser.add_argument("--model_type", default="gptsmall", type=str)
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--tokenizer_type", default="BPE", type=str)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--tied", type=bool)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--wdecay", default=1.2e-6, type=float)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--compile", default=None, choices=[None, "dynamo"])
    parser.add_argument("--implementation", default="nanogpt", choices=["nanogpt"])
    args = parser.parse_args()

    main(args)
