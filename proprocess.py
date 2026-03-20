import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter


def compute_baseline_cumulative_hazard(T, E, log_risks):
    """Compute the baseline cumulative hazard for a Cox-style model.

    Args:
        T: Array of survival times.
        E: Event indicator array (1 for event, 0 for censoring).
        log_risks: Array of individual log-risk scores.

    Returns:
        A tuple of:
            - unique_event_times: Sorted unique event times.
            - baseline_chf: Baseline cumulative hazard at each event time.
    """
    n = len(T)
    exp_risk = np.exp(log_risks)
    event_table = {}

    for i in range(n):
        if E[i] == 1:
            t = T[i]
            if t not in event_table:
                event_table[t] = 1
            else:
                event_table[t] += 1
    
    unique_event_times = np.sort(np.array(list(event_table.keys())))
    
    baseline_chf = []
    cum_hazard = 0.0
    for t in unique_event_times:
        at_risk = exp_risk[T >= t]
        d_j = event_table[t]
        R_j = np.sum(at_risk)
        delta = d_j / R_j
        cum_hazard += delta
        baseline_chf.append(cum_hazard)
    
    return unique_event_times, np.array(baseline_chf)


def predict_survival_prob(baseline_chf, log_risks):
    """Predict survival probability curves from baseline hazard and log-risks.

    Args:
        baseline_chf: Baseline cumulative hazard values over time.
        log_risks: Array of individual log-risk scores.

    Returns:
        A 2D array where each row is one individual's survival curve.
    """
    surv_probs = []
    for lr in log_risks:
        s_t = np.exp(- baseline_chf * np.exp(lr))
        surv_probs.append(s_t)
    surv_probs = np.array(surv_probs)
    return surv_probs 


def kaplan_meier_estimator(T, E):
    """Estimate a Kaplan-Meier survival curve.

    Args:
        T: Array of survival times.
        E: Event indicator array (1 for event, 0 for censoring).

    Returns:
        A tuple of:
            - unique_times: Sorted unique event times.
            - surv_prob: Kaplan-Meier survival probability at each event time.
    """
    order = np.argsort(T)
    T_sorted = T[order]
    E_sorted = E[order]
    unique_times = np.unique(T_sorted[E_sorted == 1])
    surv_prob = []
    prob = 1.0

    for t in unique_times:
        d_i = np.sum((T_sorted == t) & (E_sorted == 1))
        n_i = np.sum(T_sorted >= t)
        prob *= (1 - d_i / n_i)
        surv_prob.append(prob)

    return unique_times, np.array(surv_prob)


def calibration_curve_survival(T, E, surv_probs_pred, event_times=None, t0=10, n_bins=10, is_mean=True):
    """Compute calibration data for survival predictions at a specific time.

    Args:
        T: Array of survival times.
        E: Event indicator array.
        surv_probs_pred: Predicted survival probabilities (matrix or vector).
        event_times: Time grid corresponding to prediction columns.
        t0: Target time point for calibration.
        n_bins: Number of quantile bins.
        is_mean: If True, return mean predicted probability per bin.

    Returns:
        mean_pred_probs: Predicted survival probabilities by bin.
        mean_obs_probs: Observed survival probabilities by bin (KM-based).
        mean_vars: Approximate bin-wise variance terms.
    """
    pred_probs_at_t0 = surv_probs_pred
    if event_times is not None:
        idx = np.searchsorted(event_times, t0, side="right") - 1
        pred_probs_at_t0 = pred_probs_at_t0[:, idx]

    quantiles = np.quantile(pred_probs_at_t0, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(pred_probs_at_t0, quantiles, right=True)

    mean_pred_probs = []
    mean_obs_probs = []
    mean_vars = []
    group_masks = []
    for i in range(1, n_bins + 1):
        group_mask = bin_ids == i
        if np.sum(group_mask) == 0:
            continue
        group_T = T[group_mask]
        group_E = E[group_mask]
        group_pred = pred_probs_at_t0[group_mask]

        km_time, km_surv = kaplan_meier_estimator(group_T, group_E)
        if len(km_time) == 0 or t0 < km_time[0]:
            observed = 1.0
        else:
            observed = np.interp(t0, km_time, km_surv)
        
        if is_mean:
            mean_pred_probs.append(np.mean(group_pred))
        else:
            mean_pred_probs.append(group_pred)
        mean_obs_probs.append(observed)
        mean_vars.append(max((observed * (1 - observed)) / np.sum(group_mask), 1e-6))
        group_masks.append(group_mask)

    return mean_pred_probs, mean_obs_probs, mean_vars


def plot_calibration_curve(mean_pred_probs, mean_obs_probs, t0, save_path):
    """Plot and save the event-risk calibration curve.

    Args:
        mean_pred_probs: Bin-wise predicted survival probabilities.
        mean_obs_probs: Bin-wise observed survival probabilities.
        t0: Target calibration time point.
        save_path: Output path for the saved figure.
    """
    mean_pred_probs = [x * 100 for x in mean_pred_probs]
    mean_obs_probs = [x * 100 for x in mean_obs_probs]
    lim = int(max(mean_obs_probs + mean_pred_probs) + 5) if mean_obs_probs + mean_pred_probs else 5
    
    coeffs = np.polyfit(mean_pred_probs, mean_obs_probs, deg=1)
    poly_func = np.poly1d(coeffs)
    x_fit = np.linspace(0, lim, 200)
    y_fit = poly_func(x_fit)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, lim], [0, lim], linestyle='--', color='gray', label='Perfect Calibration')
    plt.scatter(mean_pred_probs, mean_obs_probs, color='blue', label='Observed vs Predicted', marker='o')
    plt.plot(x_fit, y_fit, linestyle='-', color='red', lw=2, label='Observed vs Predicted')
    plt.xlabel(f'Predicted Event Risk Probability(%) at t={t0}')
    plt.ylabel('Observed Event Risk Probability(%) (KM)')
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.title('Event Risk Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


def compute_c_index(time, event, risk_score):
    """Compute the concordance index (C-index).

    Args:
        time: Array of survival times.
        event: Event indicator array.
        risk_score: Risk scores where larger values indicate higher risk.

    Returns:
        The C-index value.
    """
    return concordance_index(time, -risk_score, event)


def bootstrap_c_index_ci(time, event, risk_score, n_bootstrap=1000, alpha=0.05):
    """Estimate mean C-index and confidence interval via bootstrap.

    Args:
        time: Array of survival times.
        event: Event indicator array.
        risk_score: Array of risk scores.
        n_bootstrap: Number of valid bootstrap samples to collect.
        alpha: Significance level for a two-sided confidence interval.

    Returns:
        mean_c_index: Mean bootstrap C-index.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
    """
    n_samples = len(time)
    c_indices = []
    valid_samples = 0
    max_attempts = n_bootstrap * 10 
    attempts = 0
    
    while valid_samples < n_bootstrap and attempts < max_attempts:
        try:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            c_index = compute_c_index(time[indices], event[indices], risk_score[indices])
            c_indices.append(c_index)
            valid_samples += 1
        except Exception as e:
            if "No admissable pairs" in str(e):
                attempts += 1
                continue
            else:
                raise e
    
    if valid_samples < n_bootstrap:
        print(f"warning: {valid_samples} less than {n_bootstrap}.")
    
    lower = np.percentile(c_indices, (alpha/2)*100)
    upper = np.percentile(c_indices, (1-alpha/2)*100)
    mean_c_index = np.mean(c_indices)
    
    return mean_c_index, lower, upper
