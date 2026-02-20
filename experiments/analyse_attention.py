import os
import numpy as np

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def main():
    scenario = os.environ["SCENARIO_NAME"]
    seed = int(os.environ["SEED"])
    attn_mode = os.getenv("ATTN_MODE", "").strip()

    if attn_mode:
        base = f"runs/attn_logs/{attn_mode}/{scenario}/seed{seed}"
    else:
        base = f"runs/attn_logs/{scenario}/seed{seed}"

    attn = np.load(f"{base}/attention.npy")     # (T, N)
    dx = np.load(f"{base}/dx.npy")              # (T, N)
    pres = np.load(f"{base}/presence.npy")      # (T, N)

    # ignore empty slots
    dx_masked = dx.copy()
    dx_masked[pres < 0.5] = np.nan

    # leader = nearest in front (min positive dx); if none, nearest by abs(dx)
    leader_idx = np.zeros(attn.shape[0], dtype=int)
    for t in range(attn.shape[0]):
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

    argmax_idx = np.argmax(attn, axis=1)
    hit_rate = (argmax_idx == leader_idx).mean()
    leader_weight = attn[np.arange(attn.shape[0]), leader_idx].mean()
    ent = entropy(attn).mean()

    print("T:", attn.shape[0], "N:", attn.shape[1])
    print("argmax==leader hit rate:", hit_rate)
    print("mean attention on leader:", leader_weight)
    print("mean attention entropy:", ent)

if __name__ == "__main__":
    main()
