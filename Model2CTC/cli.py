"""Command line interface for setup-time CTC design and edge export."""

from __future__ import annotations

import argparse
import json

from .config import DesignConfig
from .design import design_regularized_ctc, evaluate_ctc
from .export import export_fixed_header
from .io import load_brir, load_filter, save_filter
from .model import ResidualModel, design_hybrid_ctc
from .training import train_residual_model


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="model2ctc",
        description="Design a stable edge CTC FIR from a predicted stereo BRIR",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    design = commands.add_parser("design", help="design CTC filters from a predicted BRIR")
    design.add_argument("--brir", required=True)
    design.add_argument("--output", required=True)
    design.add_argument("--model", help="optional trained residual model")
    design.add_argument("--config")

    train = commands.add_parser("train", help="train on paired predicted/measured BRIRs")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--report")
    train.add_argument("--ridge", type=float, default=1e-2)
    train.add_argument("--config")

    evaluate = commands.add_parser("evaluate", help="evaluate a CTC filter on a measured BRIR")
    evaluate.add_argument("--brir", required=True)
    evaluate.add_argument("--filter", required=True)
    evaluate.add_argument("--config")

    export = commands.add_parser("export", help="quantize a designed filter to a C header")
    export.add_argument("--filter", required=True)
    export.add_argument("--output", required=True)

    args = parser.parse_args()
    config = DesignConfig.load(args.config) if getattr(args, "config", None) else DesignConfig()

    if args.command == "design":
        sample_rate, brir, brir_metadata = load_brir(args.brir)
        _require_rate(sample_rate, config)
        if args.model:
            filters, metadata = design_hybrid_ctc(brir, ResidualModel.load(args.model), config)
        else:
            filters, metadata = design_regularized_ctc(brir, config)
        metadata["brir_metadata"] = brir_metadata
        save_filter(args.output, sample_rate, filters, metadata)
        print(json.dumps(metadata, indent=2))
        return 0

    if args.command == "train":
        report = train_residual_model(args.manifest, args.output, config, ridge=args.ridge)
        text = json.dumps(report, indent=2) + "\n"
        if args.report:
            from pathlib import Path
            Path(args.report).write_text(text)
        print(text, end="")
        return 0

    if args.command == "evaluate":
        brir_rate, brir, _ = load_brir(args.brir)
        filter_rate, filters, _ = load_filter(args.filter)
        _require_rate(brir_rate, config)
        _require_rate(filter_rate, config)
        print(json.dumps(evaluate_ctc(brir, filters, config), indent=2))
        return 0

    if args.command == "export":
        _sample_rate, filters, _ = load_filter(args.filter)
        export_fixed_header(filters, args.output)
        print(json.dumps({"header": args.output, "taps": int(filters.shape[-1])}, indent=2))
        return 0

    return 2


def _require_rate(sample_rate: int, config: DesignConfig) -> None:
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate} Hz, got {sample_rate} Hz")


if __name__ == "__main__":
    raise SystemExit(main())
