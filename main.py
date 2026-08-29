from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CAVDE",
        description="CAVDE AI Framework",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser("train", help="Train model")
    subparsers.add_parser("chat", help="Interactive chat")
    subparsers.add_parser("benchmark", help="Run benchmark")
    subparsers.add_parser("evaluate", help="Evaluate model")
    subparsers.add_parser("export", help="Export trained model")

    return parser


def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":
        from train import main as train_main
        return train_main()

    if args.command == "chat":
        from chat import main as chat_main
        return chat_main()

    if args.command == "benchmark":

        from benchmark import benchmark

        benchmark()

        return 0
    if args.command == "export":

        from export import export

        export()

        return 0

    if args.command == "evaluate":

        from evaluate import evaluate

        evaluate()

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())