import argparse
import importlib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

from preprocess import PredictorProcessor
from models.transformer import InferAMFormer


class OnnxExportWrapper(nn.Module):
    """Wrap InferAMFormer to expose a stable ONNX export forward signature."""

    def __init__(self, model: InferAMFormer):
        """Initialize the wrapper.

        Args:
            model: Loaded InferAMFormer instance used for ONNX export inference.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        """Compute risk probability at the given event time for each sample.

        Args:
            x: Input feature tensor with shape [batch_size, num_features].
            t0: Target event time tensor with shape [1] or [batch_size].

        Returns:
            Tensor of risk probabilities aligned with the input batch.
        """
        B, N = x.shape
        n = int(self.model.num_zscore)

        if n > 0:
            mean = self.model.mean.to(device=x.device, dtype=x.dtype)
            std = self.model.std.to(device=x.device, dtype=x.dtype)
            x_norm = (x[:, :n] - mean) / std
            x = torch.cat([x_norm, x[:, n:]], dim=1) if n < N else x_norm

        x_emb = self.model.feature_tokenizer(x).view(B, N, self.model.embed_dim)
        interact_out = self.model.am_block(x_emb)
        flat_out = self.model.ffn_add(interact_out.view(B, -1))
        log_hz = self.model.mlp(x + flat_out * 0.01)

        base_hazards = self.model.base_hazards.to(device=x.device, dtype=log_hz.dtype)
        probs = 1.0 - torch.exp(-base_hazards * torch.exp(log_hz))

        t0 = t0.reshape(-1).to(device=x.device, dtype=self.model.base_event_times.dtype)
        if t0.numel() == 1:
            t0 = t0.expand(B)

        event_times = self.model.base_event_times.to(device=x.device, dtype=t0.dtype)
        idx = (t0.unsqueeze(1) >= event_times.unsqueeze(0)).to(torch.long).sum(dim=1) - 1
        idx = torch.clamp(idx, min=0, max=event_times.numel() - 1)
        batch_idx = torch.arange(B, device=x.device)
        y = probs[batch_idx, idx]
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y


def _extract_state_dict(ckpt_obj):
    """Extract a torch state_dict from common checkpoint container formats.

    Args:
        ckpt_obj: Object loaded by torch.load, usually a dict-like checkpoint.

    Returns:
        A state_dict mapping parameter names to tensors.
    """
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Unable to recognize checkpoint structure. Please check the ckpt file.")


def build_model(
    ckpt_file: str,
    model_type: str,
    embed_dim: int,
    num_prompts: int,
    top_k: int,
) -> tuple[InferAMFormer, int]:
    """Build and load InferAMFormer from checkpoint.

    Args:
        ckpt_file: Absolute or relative path to the checkpoint file.
        model_type: PredictorProcessor mode used to infer continuous predictors when needed.
        embed_dim: Fallback embedding size when it cannot be inferred from checkpoint.
        num_prompts: Number of prompt tokens used by the AMFormer block.
        top_k: Top-k routing parameter used by the AMFormer block.

    Returns:
        Tuple of (loaded model, inferred num_features).
    """
    ckpt_obj = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt_obj)

    if "feature_tokenizer.weight" not in state_dict:
        raise ValueError("Checkpoint is missing feature_tokenizer.weight, unable to infer input dimension.")

    ft_weight = state_dict["feature_tokenizer.weight"]
    if ft_weight.ndim != 2 or ft_weight.shape[1] <= 0:
        raise ValueError("feature_tokenizer.weight has an invalid shape, unable to infer num_features.")

    num_features = int(ft_weight.shape[1])
    inferred_embed_dim = int(ft_weight.shape[0] // num_features)

    if "mean" in state_dict and state_dict["mean"].numel() > 0:
        num_zscore = int(state_dict["mean"].numel())
    else:
        predictor_processor = PredictorProcessor(model_type=model_type)
        num_zscore = len(predictor_processor.con_predictors)

    if inferred_embed_dim <= 0:
        inferred_embed_dim = embed_dim

    model = InferAMFormer(
        num_features=num_features,
        num_zscore=num_zscore,
        embed_dim=inferred_embed_dim,
        num_prompts=num_prompts,
        top_k=top_k,
    ).cpu()

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, num_features


def _ensure_real_onnx_package() -> None:
    """Ensure the imported onnx module is the official package.

    This avoids name shadowing when this file is named onnx.py.
    """
    script_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_file)
    cwd = os.path.abspath(os.getcwd())

    cleaned = []
    for p in sys.path:
        rp = os.path.abspath(p) if p else cwd
        if rp not in {script_dir, cwd}:
            cleaned.append(p)
    sys.path = cleaned

    mod = sys.modules.get("onnx")
    if mod is not None and os.path.abspath(getattr(mod, "__file__", "")) == script_file:
        sys.modules.pop("onnx", None)

    onnx_pkg = importlib.import_module("onnx")
    if not hasattr(onnx_pkg, "load_model_from_string"):
        raise RuntimeError(
            "The imported onnx module is not the official package or its version is too old. Please run: pip install -U onnx"
        )


def export_onnx(
    model: InferAMFormer,
    num_features: int,
    onnx_file: str,
    opset_version: int,
    batch_size: int,
    default_t0: float,
) -> None:
    """Export the wrapped model to ONNX with dynamic axes.

    Args:
        model: Loaded InferAMFormer model to export.
        num_features: Number of model input features.
        onnx_file: Output ONNX file path.
        opset_version: ONNX opset version used during export.
        batch_size: Dummy batch size used to trace the graph.
        default_t0: Dummy event time used for export tracing.
    """
    _ensure_real_onnx_package()
    wrapper = OnnxExportWrapper(model).cpu().eval()

    dummy_x = torch.randn(batch_size, num_features, dtype=torch.float32)
    dummy_t0 = torch.tensor([default_t0], dtype=torch.float64)

    os.makedirs(os.path.dirname(onnx_file) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_x, dummy_t0),
            onnx_file,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["features", "t0"],
            output_names=["risk_prob_t0"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "t0": {0: "t0_size"},
                "risk_prob_t0": {0: "batch_size"},
            },
        )


def verify_onnx(
    model: InferAMFormer,
    onnx_file: str,
    num_features: int,
    batch_size: int,
    t0: float,
) -> None:
    """Compare ONNXRuntime output against PyTorch output.

    Args:
        model: Loaded InferAMFormer model used as PyTorch reference.
        onnx_file: Path to the exported ONNX model.
        num_features: Number of input features for random test input generation.
        batch_size: Batch size for the parity check input.
        t0: Event time value used in both PyTorch and ONNX inference.
    """
    import onnxruntime as ort

    wrapper = OnnxExportWrapper(model).cpu().eval()

    x = np.random.randn(batch_size, num_features).astype(np.float32)
    t0_np = np.array([t0], dtype=np.float64)

    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(x), torch.from_numpy(t0_np)).cpu().numpy()

    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["risk_prob_t0"], {"features": x, "t0": t0_np})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"verify max_abs_diff: {max_abs_diff:.8f}")


def main() -> None:
    """Parse CLI arguments, export ONNX model, and optionally verify parity.

    Command-line inputs are provided through argparse options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--onnx_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="Simplified")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--t0", type=float, default=10.0)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    ckpt_file = os.path.abspath(args.ckpt_file)
    onnx_file = os.path.abspath(args.onnx_file)

    model, num_features = build_model(
        ckpt_file=ckpt_file,
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        num_prompts=args.num_prompts,
        top_k=args.top_k,
    )

    export_onnx(
        model=model,
        num_features=num_features,
        onnx_file=onnx_file,
        opset_version=args.opset,
        batch_size=args.batch_size,
        default_t0=args.t0,
    )

    print(f"ONNX export completed: {onnx_file}")

    if args.verify:
        verify_onnx(
            model=model,
            onnx_file=onnx_file,
            num_features=num_features,
            batch_size=max(1, args.batch_size),
            t0=args.t0,
        )


if __name__ == "__main__":
    main()