import argparse
import time
from os import path as osp

import torch
import torch.profiler as tprof
import yaml

from attention import ATTENTION_IMPLEMENTATIONS, MultiHeadAttention


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use torch.profiler to profile attention for matmul or einsum"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.yml",
        help="Path to configuration yml file. Defaults to ./config.yml.",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="torch",
        choices=[e for e in ATTENTION_IMPLEMENTATIONS],
        help="Implementation to use for scaled dot product attention (softmax((QK^T)/scale_factor)V).",
    )
    parser.add_argument(
        "--save_trace",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to save torch profile as a .json trace. Defaults to False.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./traces",
        help="Directory to write output trace file in. Only written to if save_trace is set. Defaults to ./traces",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="trace.json",
        help="Name of output trace file. Only written to if `save_trace` is set. Defaults to trace.json.",
    )
    args = parser.parse_args()
    assert args.attn_impl in ATTENTION_IMPLEMENTATIONS
    return args


def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    attn_dims = config["embedding_dims"] // config["num_heads"]
    assert attn_dims * config["num_heads"] == config["embedding_dims"]

    device = torch.device("cuda")

    X = torch.randn(
        (
            config["num_iters"],
            config["batch_size"],
            config["sequence_length"],
            config["embedding_dims"],
        )
    ).to(device)

    attention = MultiHeadAttention(
        config["embedding_dims"],
        attn_dims,
        config["num_heads"],
        attn_impl=args.attn_impl,
    ).to(device)

    activities = [tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA]

    with tprof.profile(activities=activities) as prof:
        for x in X:
            attention(x)

    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    if args.save_trace:
        outfile = osp.join(args.output_dir, args.output_file)
        print(f"Writing trace to: {outfile}")
        prof.export_chrome_trace(outfile)


if __name__ == "__main__":
    main(parse_args())
