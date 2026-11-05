import os
import numpy as np

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def compute_leader_idx(dx: np.ndarray, pres: np.ndarray) -> np.ndarray:
    """Nearest in front; if none, nearest by |dx|."""
    leader_idx = np.zeros(dx.shape[0], dtype=int)
    for t in range(dx.shape[0]):
        valid = pres[t] >= 0.5
        if not valid.any():
            leader_idx[t] = 0
            continue
        dx_t = dx[t].copy()
        dx_t[~valid] = np.inf

        front = (dx_t > 0) & np.isfinite(dx_t)
        if front.any():
            leader_idx[t] = int(np.argmin(np.where(front, dx_t, np.inf)))
        else:
            leader_idx[t] = int(np.argmin(np.abs(dx_t)))
    return leader_idx


def compute_ttc_idx(dx: np.ndarray, vx: np.ndarray, pres: np.ndarray, eps: float = 1e-6):
    """
    Longitudinal TTC proxy from relative features.
    Closing if dx and dvx have opposite signs (dx * dvx < 0).
    Falls back to leader heuristic if no closing neighbour exists.
    """
    ttc_idx = np.zeros(dx.shape[0], dtype=int)
    has_closing = np.zeros(dx.shape[0], dtype=bool)
    min_ttc = np.full(dx.shape[0], np.nan, dtype=np.float32)

    leader_fallback = compute_leader_idx(dx, pres)

    for t in range(dx.shape[0]):
        valid = pres[t] >= 0.5
        if not valid.any():
            ttc_idx[t] = 0
            continue

        dx_t = dx[t]
        vx_t = vx[t]
        closing = valid & ((dx_t * vx_t) < 0.0)
        if not closing.any():
            ttc_idx[t] = leader_fallback[t]
            continue

        ttc = np.full(dx_t.shape, np.inf, dtype=np.float32)
        ttc[closing] = np.abs(dx_t[closing]) / (np.abs(vx_t[closing]) + eps)
        idx = int(np.argmin(ttc))

        ttc_idx[t] = idx
        has_closing[t] = True
        min_ttc[t] = float(ttc[idx])

    return ttc_idx, has_closing, min_ttc

def main():
    scenario = os.environ["SCENARIO_NAME"]
    exp_version = os.getenv("EXP_VERSION", "").strip()
    seed = int(os.environ["SEED"])
    attn_mode = os.getenv("ATTN_MODE", "").strip()

    if exp_version and attn_mode:
        base = f"runs/attn_logs/{exp_version}/{attn_mode}/{scenario}/seed{seed}"
    elif exp_version:
        base = f"runs/attn_logs/{exp_version}/{scenario}/seed{seed}"
    elif attn_mode:
        base = f"runs/attn_logs/{attn_mode}/{scenario}/seed{seed}"
    else:
        base = f"runs/attn_logs/{scenario}/seed{seed}"

    attn = np.load(f"{base}/attention.npy")     # (T, N)
    dx = np.load(f"{base}/dx.npy")              # (T, N)
    pres = np.load(f"{base}/presence.npy")      # (T, N)
    vx_path = f"{base}/vx.npy"
    vx = np.load(vx_path) if os.path.exists(vx_path) else None

    # ignore empty slots
    dx_masked = dx.copy()
    dx_masked[pres < 0.5] = np.nan

    argmax_idx = np.argmax(attn, axis=1)
    leader_idx = compute_leader_idx(dx, pres)
    hit_rate = (argmax_idx == leader_idx).mean()
    leader_weight = attn[np.arange(attn.shape[0]), leader_idx].mean()
    ent = entropy(attn).mean()

    print("T:", attn.shape[0], "N:", attn.shape[1])
    print("argmax==leader hit rate:", hit_rate)
    print("mean attention on leader:", leader_weight)
    print("mean attention entropy:", ent)

    if vx is None:
        print("TTC metrics unavailable: missing vx.npy (rerun eval_attention.py with updated logger).")
        return

    ttc_idx, has_closing, min_ttc = compute_ttc_idx(dx, vx, pres)
    ttc_hit_rate = (argmax_idx == ttc_idx).mean()
    ttc_weight = attn[np.arange(attn.shape[0]), ttc_idx].mean()
    closing_frac = has_closing.mean()
    finite_ttc = np.isfinite(min_ttc)
    mean_min_ttc = float(np.nanmean(min_ttc[finite_ttc])) if finite_ttc.any() else float("nan")

    print("argmax==ttc_critical hit rate:", ttc_hit_rate)
    print("mean attention on ttc_critical:", ttc_weight)
    print("fraction steps with closing TTC candidate:", closing_frac)
    print("mean min TTC (closing steps only):", mean_min_ttc)

if __name__ == "__main__":
    main()
