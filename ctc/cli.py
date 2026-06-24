from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_pipeline
from .pipeline import init_configs, render_stage1, render_stage2, run_pipeline
from .training import train_linear_residual_model


def main() -> int:
    parser = argparse.ArgumentParser(prog="ctc", description="ML-guided crosstalk cancellation reference CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="write default geometry, room, and runtime JSON")
    init_parser.add_argument("--directory", default="configs")

    stage1_parser = subparsers.add_parser("stage1", help="render crosstalk-cancelled loudspeaker feeds")
    _add_common_render_args(stage1_parser)
    stage1_parser.add_argument("input")
    stage1_parser.add_argument("--output", default="preprocessed.wav")
    stage1_parser.add_argument("--room")
    stage1_parser.add_argument("--model")
    stage1_parser.add_argument("--pcm16", action="store_true")

    stage2_parser = subparsers.add_parser("stage2", help="simulate binaural playback of loudspeaker feeds")
    _add_common_render_args(stage2_parser)
    stage2_parser.add_argument("input")
    stage2_parser.add_argument("--output", default="simulated_binaural.wav")
    stage2_parser.add_argument("--pcm16", action="store_true")

    run_parser = subparsers.add_parser("run", help="run Stage 1 and Stage 2")
    _add_common_render_args(run_parser)
    run_parser.add_argument("input")
    run_parser.add_argument("--stage1-output", default="preprocessed.wav")
    run_parser.add_argument("--stage2-output", default="simulated_binaural.wav")
    run_parser.add_argument("--room")
    run_parser.add_argument("--model")

    train_parser = subparsers.add_parser("train", help="train a dependency-free ML residual controller")
    train_parser.add_argument("--output", default="ml_filter_model.json")
    train_parser.add_argument("--examples", type=int, default=192)
    train_parser.add_argument("--ridge", type=float, default=1e-3)

    eval_parser = subparsers.add_parser("evaluate", help="write comprehensive audio and CTC metrics")
    _add_common_render_args(eval_parser)
    eval_parser.add_argument("--input", required=True)
    eval_parser.add_argument("--stage1", required=True)
    eval_parser.add_argument("--stage2", required=True)
    eval_parser.add_argument("--room")
    eval_parser.add_argument("--model")
    eval_parser.add_argument("--output", default="metrics.json")

    args = parser.parse_args()
    if args.command == "init-config":
        init_configs(args.directory)
        print(json.dumps({"configs": str(Path(args.directory).resolve())}, indent=2))
        return 0
    if args.command == "stage1":
        metrics = render_stage1(
            args.input,
            args.output,
            geometry_path=args.geometry,
            room_path=args.room,
            runtime_path=args.runtime,
            model_path=args.model,
            pcm16=args.pcm16,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "stage2":
        metrics = render_stage2(
            args.input,
            args.output,
            geometry_path=args.geometry,
            runtime_path=args.runtime,
            pcm16=args.pcm16,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "run":
        metrics = run_pipeline(
            args.input,
            stage1_output=args.stage1_output,
            stage2_output=args.stage2_output,
            geometry_path=args.geometry,
            room_path=args.room,
            runtime_path=args.runtime,
            model_path=args.model,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "train":
        metrics = train_linear_residual_model(
            output_path=args.output,
            examples=args.examples,
            ridge=args.ridge,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "evaluate":
        metrics = evaluate_pipeline(
            input_path=args.input,
            stage1_path=args.stage1,
            stage2_path=args.stage2,
            geometry_path=args.geometry,
            room_path=args.room,
            runtime_path=args.runtime,
            model_path=args.model,
            output_path=args.output,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


def _add_common_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--geometry")
    parser.add_argument("--runtime")


if __name__ == "__main__":
    raise SystemExit(main())
