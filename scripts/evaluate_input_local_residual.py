from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from evaluate_additive_holdout import collect


def relative_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def fit_local_model(x: np.ndarray, y: np.ndarray, clusters: int, rank: int, ridge: float, seed: int) -> dict:
    """Fit y ~= base[k] + U[k] @ (V[k].T @ (x - center[k]))."""
    actual_clusters = min(clusters, len(x))
    km = MiniBatchKMeans(
        n_clusters=actual_clusters,
        random_state=seed,
        n_init=4,
        batch_size=min(512, len(x)),
        max_iter=200,
    )
    code = km.fit_predict(x)
    dim = x.shape[1]
    centers = np.zeros((actual_clusters, dim), dtype=np.float32)
    bases = np.zeros((actual_clusters, y.shape[1]), dtype=np.float32)
    input_bases: list[np.ndarray] = []
    output_bases: list[np.ndarray] = []

    for k in range(actual_clusters):
        member = np.flatnonzero(code == k)
        if len(member) == 0:
            centers[k] = x.mean(axis=0)
            bases[k] = y.mean(axis=0)
            input_bases.append(np.zeros((dim, 0), dtype=np.float32))
            output_bases.append(np.zeros((y.shape[1], 0), dtype=np.float32))
            continue

        xk, yk = x[member], y[member]
        center, base = xk.mean(axis=0), yk.mean(axis=0)
        dx, dy = xk - center, yk - base
        centers[k], bases[k] = center, base

        # The data rank caps the useful local residual rank.  This factorization
        # maps the CPU-produced coefficients directly onto a GPU output basis.
        useful_rank = min(rank, len(member) - 1, dim)
        if useful_rank <= 0:
            input_bases.append(np.zeros((dim, 0), dtype=np.float32))
            output_bases.append(np.zeros((y.shape[1], 0), dtype=np.float32))
            continue
        _, _, vt = np.linalg.svd(dx, full_matrices=False)
        v = vt[:useful_rank].T.astype(np.float32, copy=False)
        coeff = dx @ v
        gram = coeff.T @ coeff
        coef_to_output = np.linalg.solve(
            gram + np.eye(useful_rank, dtype=np.float32) * ridge,
            coeff.T @ dy,
        )
        input_bases.append(v)
        output_bases.append(coef_to_output.T.astype(np.float32, copy=False))

    return {"centers": centers, "bases": bases, "input_bases": input_bases, "output_bases": output_bases}


def predict(x: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    centers = model["centers"]
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
    code = np.argmin(d2, axis=1)
    distance = np.sqrt(np.take_along_axis(d2, code[:, None], axis=1)[:, 0])
    out = model["bases"][code].copy()
    coefficients: list[np.ndarray] = []
    for k in range(len(centers)):
        rows = np.flatnonzero(code == k)
        v, u = model["input_bases"][k], model["output_bases"][k]
        if len(rows) == 0:
            continue
        c = (x[rows] - centers[k]) @ v
        out[rows] += c @ u.T
        coefficients.append(c)
    return out, code, distance, coefficients


def pick_threshold(distance: np.ndarray, err: np.ndarray, p95_budget: float) -> tuple[float | None, float, float | None]:
    order = np.argsort(distance)
    best: tuple[float | None, float, float | None] = (None, 0.0, None)
    for n in range(1, len(order) + 1):
        accepted = err[order[:n]]
        p95 = float(np.percentile(accepted, 95))
        if p95 <= p95_budget:
            best = (float(distance[order[n - 1]]), n / len(order), p95)
    return best


def routed_metrics(distance: np.ndarray, err: np.ndarray, threshold: float | None) -> dict[str, float]:
    accept = np.zeros(len(distance), dtype=bool) if threshold is None else distance <= threshold
    accepted = err[accept]
    return {
        "approx_fraction": float(accept.mean()),
        "fallback_fraction": float((~accept).mean()),
        "accepted_rel_l2": float(accepted.mean()) if len(accepted) else 0.0,
        "accepted_rel_l2_p95": float(np.percentile(accepted, 95)) if len(accepted) else 0.0,
        # Exact fallbacks contribute no output error in the hybrid path.
        "hybrid_rel_l2": float(accepted.sum() / len(err)) if len(err) else 0.0,
    }


def profile_cpu_route(x: np.ndarray, model: dict, repeats: int) -> dict[str, float]:
    # This intentionally times only CPU classification/coefficient generation.
    # PCIe and CUDA synchronization require the later graph-callback prototype.
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        predict(x, model)
        timings.append((time.perf_counter_ns() - start) / len(x))
    return {"per_token_us_median": float(np.median(timings) / 1000), "per_token_us_p95": float(np.percentile(timings, 95) / 1000)}


def model_bytes(model: dict) -> dict[str, int]:
    centers = model["centers"].nbytes
    bases = model["bases"].nbytes
    v = sum(item.nbytes for item in model["input_bases"])
    u = sum(item.nbytes for item in model["output_bases"])
    return {
        "cpu_centers_and_v_bytes": centers + v,
        "gpu_bases_and_u_bytes": bases + u,
        "all_artifact_bytes": centers + bases + v + u,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-routed local low-rank FFN residual experiment")
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--budgets", default="0.10,0.20,0.30,0.50")
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train, holdout = collect(args.calibration_root), collect(args.holdout_root)
    if args.layer not in train or args.layer not in holdout:
        raise SystemExit(f"layer {args.layer} was not captured in both datasets")
    train_x, _, train_y = train[args.layer]
    test_x, _, test_y = holdout[args.layer]
    ranks = [int(item) for item in args.ranks.split(",") if item.strip()]
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    rows = []
    for rank in ranks:
        model = fit_local_model(train_x, train_y, args.clusters, rank, args.ridge, 23000 + rank)
        train_pred, _, train_dist, _ = predict(train_x, model)
        test_pred, _, test_dist, _ = predict(test_x, model)
        train_err, test_err = relative_l2(train_pred, train_y), relative_l2(test_pred, test_y)
        policies = []
        for budget in budgets:
            threshold, coverage, actual = pick_threshold(train_dist, train_err, budget)
            policies.append({
                "train_p95_budget": budget,
                "distance_threshold": threshold,
                "train_approx_fraction": coverage,
                "train_accepted_rel_l2_p95": actual,
                "holdout": routed_metrics(test_dist, test_err, threshold),
            })
        rows.append({
            "rank": rank,
            "full_approximation": {
                "train_rel_l2": float(train_err.mean()),
                "holdout_rel_l2": float(test_err.mean()),
                "holdout_rel_l2_p95": float(np.percentile(test_err, 95)),
                "distance_error_corr": float(np.corrcoef(test_dist, test_err)[0, 1]),
            },
            "artifact": model_bytes(model),
            "cpu_route_profile": profile_cpu_route(test_x, model, args.profile_repeats),
            "policies": policies,
        })
    result = {
        "experiment": "input_routed_local_low_rank_ffn",
        "formula": "y_hat = base[k] + U[k] @ (V[k].T @ (x - center[k]))",
        "runtime_design": {
            "cpu": "copy FFN input x to RAM, choose nearest center, and form rank-r coefficients",
            "gpu": "keep base[k] and U[k] resident; receive only code/coefficients for approximate tokens",
            "fallback": "run exact FFN when x-center distance exceeds the calibration threshold",
            "per_approx_token_transfer_bytes": {"d2h_input_f32": int(train_x.shape[1] * 4), "h2d_coeff_f32_rank_r": "4 * rank", "h2d_base": 0},
        },
        "layer": args.layer,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "clusters": args.clusters,
        "ridge": args.ridge,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
