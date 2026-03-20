import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import pickle
from pprint import pprint
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from preprocess import PredictorProcessor
from proprocess import compute_baseline_cumulative_hazard, predict_survival_prob, compute_c_index
from utils import Surv_dataset
# from arguments import parse_args
from models.transformer import TrainAMFormer

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stages_infer(data_file, ckpt_file, sex, model_type='Simplified', t0=10, output_csv=None):
    """Run inference and save sample-level risk scores to a CSV file.

    Args:
        data_file: Path to the input data file.
        ckpt_file: Path to the model checkpoint file.
        sex: Sex filter value, 0 or 1.
        model_type: Model variant, Simplified or Full.
        t0: Target time point.
        output_csv: Output CSV path. If None, a default path under ckpt directory is used.

    Returns:
        A tuple of (c_index, output_csv).
    """
    predictor_processor = PredictorProcessor(model_type=model_type)
    surv_df = predictor_processor(data_file)
    surv_df = surv_df[surv_df['Sex'] == sex].reset_index(drop=True).drop(columns=['Sex'])
    con_columns = predictor_processor.con_predictors
    num_features = predictor_processor.num_predictors

    mean_file = os.path.join(os.path.dirname(ckpt_file), 'mean.npy')
    std_file = os.path.join(os.path.dirname(ckpt_file), 'std.npy')
    if not os.path.exists(mean_file) or not os.path.exists(std_file):
        raise FileNotFoundError(f"mean.npy or std.npy not found in {os.path.dirname(ckpt_file)}")
    zscore_mean = np.load(mean_file)
    zscore_std = np.load(std_file)
    surv_df[con_columns] = (surv_df[con_columns] - zscore_mean) / zscore_std

    Surv_dataloader = DataLoader(
        Surv_dataset(
            torch.tensor(surv_df.iloc[:, :-2].values, dtype=torch.float32).to(device),
            torch.tensor(surv_df.iloc[:, -2].values, dtype=torch.float32).to(device),
            torch.tensor(surv_df.iloc[:, -1].values, dtype=torch.float32).to(device)),
        batch_size=len(surv_df),
        shuffle=False
    )

    cox_model = TrainAMFormer(num_features=num_features, embed_dim=128, num_prompts=16, top_k=4).to(device)
    cox_model.load_state_dict(torch.load(ckpt_file, weights_only=False))
    cox_model.eval()

    with torch.no_grad():
        features, (event, timestamp) = next(iter(Surv_dataloader))
        log_hz = cox_model(features).cpu().numpy().squeeze()
        event = event.cpu().numpy().squeeze()
        timestamp = timestamp.cpu().numpy().squeeze()

        event_file_path = os.path.join(os.path.dirname(ckpt_file), 'base_event_times.npy')
        hazard_file_path = os.path.join(os.path.dirname(ckpt_file), 'base_hazards.npy')
        if os.path.exists(event_file_path) and os.path.exists(hazard_file_path):
            base_event_times = np.load(event_file_path)
            baseline_hazards = np.load(hazard_file_path)
        else:
            base_event_times, baseline_hazards = compute_baseline_cumulative_hazard(timestamp, event, log_hz)
            np.save(event_file_path, base_event_times)
            np.save(hazard_file_path, baseline_hazards)
        risk = 1.0 - predict_survival_prob(baseline_hazards, log_hz)[:, np.where(base_event_times == t0)[0][0]]

        c_index = compute_c_index(timestamp, event, risk)

        if output_csv is None:
            output_csv = os.path.join(os.path.dirname(ckpt_file), f'infer_results_sex_{sex}_t_{t0}.csv')
        output_csv = os.path.abspath(output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        result_df = pd.DataFrame({
            'event': event.astype(np.int32),
            'time': timestamp,
            'risk': risk,
        })
        result_df.to_csv(output_csv, index=False)

        print(c_index)
        print(f'Prediction results saved to: {output_csv}')
        return c_index, output_csv


def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--sex", type=int, default=0, choices=[0, 1])
    parser.add_argument("--model_type", type=str, default='Simplified', choices=['Simplified', 'Full'])
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--t0", type=float, default=10)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_file = args.data_file
    sex = args.sex
    model_type = args.model_type
    ckpt_file = args.ckpt_file

    stages_infer(
        data_file=args.data_file,
        ckpt_file=ckpt_file,
        sex=sex,
        model_type=model_type,
        t0=args.t0,
        output_csv=args.output_csv,
    )
