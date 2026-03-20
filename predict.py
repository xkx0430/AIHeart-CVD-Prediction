import argparse
import os

import numpy as np
import pandas as pd
from onnxruntime import InferenceSession

from preprocess import PredictorProcessor
from proprocess import compute_c_index


def _load_data(data_file: str, model_type: str, sex_value: int):
    """Load and preprocess data, then filter by sex and return model-ready arrays.

    Args:
        data_file: Path to the input data file.
        model_type: Preprocessing mode.
        sex_value: Sex value used for filtering.

    Returns:
        features: Feature matrix in float32.
        event: Event indicator array in bool.
        timestamp: Follow-up time array in float32.
    """
    predictor_processor = PredictorProcessor(model_type=model_type)
    surv_df = predictor_processor(data_file)
    if surv_df is None:
        raise ValueError(f"数据处理失败: {data_file}")
    surv_df = surv_df[surv_df["Sex"] == sex_value].reset_index(drop=True).drop(columns=["Sex"])
    if len(surv_df) == 0:
        raise ValueError(f"过滤 Sex={sex_value} 后无样本")

    features = surv_df.iloc[:, :-2].values.astype(np.float32)
    event = surv_df.iloc[:, -2].values.astype(bool)
    timestamp = surv_df.iloc[:, -1].values.astype(np.float32)
    return features, event, timestamp


def _build_feed_dict(session: InferenceSession, features: np.ndarray, t0: float):
    """Build ONNX inference inputs for a single sample.

    Prefer input names "features" and "t0" when available; otherwise
    fallback to the first two model inputs by position.
    """
    inputs = session.get_inputs()
    if len(inputs) < 2:
        raise ValueError("ONNX 模型输入少于2个，期望包含 features 和 t0")

    input_names = [x.name for x in inputs]
    if "features" in input_names:
        features_name = "features"
    else:
        features_name = input_names[0]

    if "t0" in input_names:
        t0_name = "t0"
    else:
        t0_name = input_names[1]

    t0_arr = np.array([t0], dtype=np.float64)
    return {features_name: features, t0_name: t0_arr}


def main():
    """CLI entry point for loading data, running inference, scoring, and saving results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="Simplified")
    parser.add_argument("--sex", type=int, default=0)
    parser.add_argument("--t0", type=float, default=10.0)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--print_head", type=int, default=100)
    args = parser.parse_args()

    onnx_file = os.path.abspath(args.onnx_file)
    features, event, timestamp = _load_data(args.data_file, args.model_type, args.sex)

    session = InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

    output_name = session.get_outputs()[0].name
    risks = []
    for i in range(features.shape[0]):
        single_features = features[i:i + 1]
        single_feed_dict = _build_feed_dict(session, single_features, args.t0)
        single_risk = session.run([output_name], single_feed_dict)[0]
        single_risk = float(np.asarray(single_risk).reshape(-1)[0])
        risks.append(single_risk)
    risk = np.asarray(risks, dtype=np.float32)

    # head_n = max(0, min(args.print_head, risk.shape[0]))
    # with np.printoptions(threshold=np.inf):
    #     print(risk[:head_n])

    c_index = compute_c_index(timestamp, event, risk)
    print(c_index)

    if args.output_csv is None:
        args.output_csv = os.path.join(
            os.path.dirname(onnx_file),
            f"predict_results_sex_{args.sex}_t_{args.t0}.csv",
        )
    output_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    result_df = pd.DataFrame({
        "event": event.astype(np.int32),
        "time": timestamp,
        "risk": risk,
    })
    result_df.to_csv(output_csv, index=False)
    print(f"Prediction results saved to: {output_csv}")


if __name__ == "__main__":
    main()
