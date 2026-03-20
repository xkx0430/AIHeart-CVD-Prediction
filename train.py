import argparse
import copy
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from loss import neg_partial_log_likelihood
from models.transformer import TrainAMFormer
from preprocess import PredictorProcessor
from proprocess import compute_baseline_cumulative_hazard, compute_c_index
from utils import Surv_dataset


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n_samples: int, val_ratio: float, seed: int):
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to split train/validation sets.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    val_size = int(round(n_samples * val_ratio))
    val_size = min(max(val_size, 1), n_samples - 1)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def prepare_data(data_file: str, model_type: str, sex: int, val_ratio: float, seed: int):
    predictor_processor = PredictorProcessor(model_type=model_type)
    surv_df = predictor_processor(data_file)
    if surv_df is None:
        raise ValueError(f"Data preprocessing failed: {data_file}")
    if sex >= 0:
        surv_df = surv_df[surv_df["Sex"] == sex].reset_index(drop=True)
    if len(surv_df) < 2:
        raise ValueError(f"Insufficient samples after filtering: {len(surv_df)}")
    surv_df = surv_df.drop(columns=["Sex"]).reset_index(drop=True)

    feature_cols = list(surv_df.columns[:-2])
    con_columns = [c for c in predictor_processor.con_predictors if c in feature_cols]
    event_col = surv_df.columns[-2]
    time_col = surv_df.columns[-1]

    train_idx, val_idx = split_indices(len(surv_df), val_ratio=val_ratio, seed=seed)
    train_df = surv_df.iloc[train_idx].copy()
    val_df = surv_df.iloc[val_idx].copy()
    full_df = surv_df.copy()

    if len(con_columns) > 0:
        zscore_mean = train_df[con_columns].mean().to_numpy(dtype=np.float32)
        zscore_std = train_df[con_columns].std(ddof=0).to_numpy(dtype=np.float32)
        zscore_std[zscore_std == 0] = 1.0
        train_df[con_columns] = ((train_df[con_columns].to_numpy(dtype=np.float32) - zscore_mean) / zscore_std).astype(np.float32)
        val_df[con_columns] = ((val_df[con_columns].to_numpy(dtype=np.float32) - zscore_mean) / zscore_std).astype(np.float32)
        full_df[con_columns] = ((full_df[con_columns].to_numpy(dtype=np.float32) - zscore_mean) / zscore_std).astype(np.float32)
    else:
        zscore_mean = np.array([], dtype=np.float32)
        zscore_std = np.array([], dtype=np.float32)

    train_x = train_df[feature_cols].to_numpy(dtype=np.float32)
    train_event = train_df[event_col].to_numpy(dtype=np.float32)
    train_time = train_df[time_col].to_numpy(dtype=np.float32)

    val_x = val_df[feature_cols].to_numpy(dtype=np.float32)
    val_event = val_df[event_col].to_numpy(dtype=np.float32)
    val_time = val_df[time_col].to_numpy(dtype=np.float32)

    full_x = full_df[feature_cols].to_numpy(dtype=np.float32)
    full_event = full_df[event_col].to_numpy(dtype=np.float32)
    full_time = full_df[time_col].to_numpy(dtype=np.float32)

    return {
        "train": (train_x, train_event, train_time),
        "val": (val_x, val_event, val_time),
        "full": (full_x, full_event, full_time),
        "num_features": len(feature_cols),
        "zscore_mean": zscore_mean,
        "zscore_std": zscore_std,
    }


def build_loader(features: np.ndarray, events: np.ndarray, times: np.ndarray, batch_size: int, shuffle: bool):
    dataset = Surv_dataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(events, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model: TrainAMFormer, loader: DataLoader, optimizer=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    events = []
    times = []

    for features, (event, timestamp) in loader:
        features = features.to(device)
        event = event.to(device)
        timestamp = timestamp.to(device)
        log_hz = model(features).reshape(-1)
        loss = neg_partial_log_likelihood(log_hz, event, timestamp)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        preds.append(log_hz.detach().cpu().numpy())
        events.append(event.detach().cpu().numpy())
        times.append(timestamp.detach().cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    event_np = np.concatenate(events, axis=0)
    time_np = np.concatenate(times, axis=0)
    c_index = compute_c_index(time_np, event_np, pred_np)
    return float(np.mean(losses)), float(c_index)


def ensure_t0_in_baseline(base_event_times: np.ndarray, base_hazards: np.ndarray, t0: float):
    if np.any(np.isclose(base_event_times, t0)):
        idx = int(np.argmin(np.abs(base_event_times - t0)))
        base_event_times[idx] = t0
        return base_event_times, base_hazards
    insert_idx = int(np.searchsorted(base_event_times, t0, side="left"))
    if insert_idx == 0:
        h_t0 = 0.0
    else:
        h_t0 = float(base_hazards[insert_idx - 1])
    new_times = np.insert(base_event_times, insert_idx, t0)
    new_hazards = np.insert(base_hazards, insert_idx, h_t0)
    return new_times, new_hazards


def train(args):
    set_seed(args.seed)
    packed = prepare_data(
        data_file=args.data_file,
        model_type=args.model_type,
        sex=args.sex,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_x, train_event, train_time = packed["train"]
    val_x, val_event, val_time = packed["val"]
    full_x, full_event, full_time = packed["full"]
    num_features = packed["num_features"]
    zscore_mean = packed["zscore_mean"]
    zscore_std = packed["zscore_std"]

    train_loader = build_loader(train_x, train_event, train_time, args.batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_event, val_time, max(1, len(val_x)), shuffle=False)

    model = TrainAMFormer(
        num_features=num_features,
        embed_dim=args.embed_dim,
        num_prompts=args.num_prompts,
        top_k=args.top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_c = run_epoch(model, train_loader, optimizer=optimizer)
        with torch.no_grad():
            val_loss, val_c = run_epoch(model, val_loader, optimizer=None)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        print(
            f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_c={train_c:.6f} | val_c={val_c:.6f}"
        )
        if wait >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(args.ckpt_dir, args.ckpt_name)
    torch.save(model.state_dict(), ckpt_file)
    np.save(os.path.join(args.ckpt_dir, "mean.npy"), zscore_mean)
    np.save(os.path.join(args.ckpt_dir, "std.npy"), zscore_std)

    with torch.no_grad():
        full_tensor = torch.tensor(full_x, dtype=torch.float32, device=device)
        full_log_hz = model(full_tensor).detach().cpu().numpy().reshape(-1)
    base_event_times, base_hazards = compute_baseline_cumulative_hazard(full_time, full_event, full_log_hz)
    base_event_times, base_hazards = ensure_t0_in_baseline(base_event_times, base_hazards, args.t0)
    np.save(os.path.join(args.ckpt_dir, "base_event_times.npy"), base_event_times)
    np.save(os.path.join(args.ckpt_dir, "base_hazards.npy"), base_hazards)

    full_c = compute_c_index(full_time, full_event, full_log_hz)
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"full_c_index={full_c:.6f}")
    print(f"checkpoint_saved={os.path.abspath(ckpt_file)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="Simplified", choices=["Simplified", "Full"])
    parser.add_argument("--sex", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="transformer_model_t0.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--t0", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
