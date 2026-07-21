"""Train the low-rank residual from paired predicted and measured BRIRs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .config import DesignConfig
from .design import design_regularized_ctc, evaluate_ctc, limit_filter_gain
from .io import load_brir
from .model import ResidualModel, brir_features


PathLike = Union[str, Path]


def train_residual_model(
    manifest_path: PathLike,
    output_path: PathLike,
    config: DesignConfig,
    ridge: float = 1e-2,
) -> Dict[str, Any]:
    cases = _load_cases(manifest_path, config)
    train_cases = [case for case in cases if case[0] != "test"]
    if len(train_cases) < 2:
        raise ValueError("at least two non-test BRIR pairs are required")

    features = np.stack([case[2] for case in train_cases])
    residual_paths = np.concatenate([case[3].reshape(4, config.filter_taps) for case in train_cases])
    _u, _s, vh = np.linalg.svd(residual_paths, full_matrices=False)
    component_count = min(config.residual_components, vh.shape[0])
    basis = vh[:component_count]

    targets = []
    for _split, _case_id, _features, residual, _measured, _base in train_cases:
        targets.append((residual.reshape(4, config.filter_taps) @ basis.T).ravel())
    targets_array = np.stack(targets)

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    x = np.column_stack((np.ones(len(features)), (features - feature_mean) / feature_scale))
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ targets_array)

    model = ResidualModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        basis=basis,
        metadata={
            "training_cases": len(train_cases),
            "ridge": ridge,
            "residual_components": component_count,
            "filter_taps": config.filter_taps,
            "feature_bins": config.feature_bins,
        },
    )
    model.save(output_path)
    report = _evaluate_cases(cases, model, config)
    report.update(model.metadata)
    return report


def _load_cases(manifest_path: PathLike, config: DesignConfig) -> List[Tuple[Any, ...]]:
    manifest_file = Path(manifest_path)
    data = json.loads(manifest_file.read_text())
    root = manifest_file.parent
    loaded = []
    for index, case in enumerate(data.get("cases", [])):
        predicted_rate, predicted, _ = load_brir(root / case["predicted_brir"])
        measured_rate, measured, _ = load_brir(root / case["measured_brir"])
        if predicted_rate != config.sample_rate or measured_rate != config.sample_rate:
            raise ValueError(f"case {case.get('id', index)} sample rate mismatch")
        base, _ = design_regularized_ctc(predicted, config)
        teacher, _ = design_regularized_ctc(measured, config)
        loaded.append((
            case.get("split", "train"),
            case.get("id", str(index)),
            brir_features(predicted, config),
            teacher - base,
            measured,
            base,
        ))
    return loaded


def _evaluate_cases(cases: List[Tuple[Any, ...]], model: ResidualModel, config: DesignConfig) -> Dict[str, Any]:
    rows = []
    for split, case_id, features, _residual, measured, base in cases:
        normalized = (features - model.feature_mean) / model.feature_scale
        coefficients = np.concatenate(([1.0], normalized)) @ model.weights
        correction = (coefficients.reshape(4, model.basis.shape[0]) @ model.basis).reshape(base.shape)
        hybrid, _ = limit_filter_gain(base + correction, config)
        rows.append({
            "id": case_id,
            "split": split,
            "baseline": evaluate_ctc(measured, base, config),
            "hybrid": evaluate_ctc(measured, hybrid, config),
        })
    test_rows = [row for row in rows if row["split"] == "test"] or rows
    return {
        "evaluation_cases": rows,
        "mean_baseline_suppression_db": float(np.mean([
            row["baseline"]["median_crosstalk_suppression_db"] for row in test_rows
        ])),
        "mean_hybrid_suppression_db": float(np.mean([
            row["hybrid"]["median_crosstalk_suppression_db"] for row in test_rows
        ])),
    }
