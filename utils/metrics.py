import numpy as np
import pandas as pd

def compute_prequential_accuracy(df, window_size=500, step=500):
    df = df.copy()
    df['correct'] = df['label'] == df['prediction']
    rolling = df['correct'].rolling(window=window_size, step=step, min_periods=window_size).mean()
    # print(f"[metrics_utils] Computed prequential accuracy (mean={rolling.mean():.4f})")
    return rolling



def analyze_degradations(predictions, partial_size=500, degrade_window=5, degrade_factor=0.95):

    partials = []
    durations = []

    correct = 0
    in_degrade = False
    current_d = 0
    episode_threshold = None

    for i, (t, p) in enumerate(predictions, 1):
        if t == p:
            correct += 1

        if i % partial_size == 0:
            acc = correct / partial_size
            partials.append(acc)
            correct = 0

            if not in_degrade:
                if len(partials) > degrade_window:
                    baseline = np.mean(partials[-(degrade_window+1):-1])
                else:
                    baseline = acc
                threshold = degrade_factor * baseline

                if acc < threshold:
                    in_degrade = True
                    episode_threshold = threshold
                    current_d = 1
            else:
                if acc < episode_threshold:
                    current_d += 1
                else:
                    durations.append(current_d)
                    in_degrade = False
                    episode_threshold = None
                    current_d = 0

    if in_degrade:
        durations.append(current_d)

    total_windows = len(partials)
    num_deg = len(durations)
    frac = sum(durations) / total_windows if total_windows else 0
    per100 = (num_deg / total_windows) * 100 if total_windows else 0
    avg_rec = np.mean(durations) if durations else 0.0

    # print(f"[metrics_utils] Degradations: {num_deg}, per100: {per100:.2f}, avg_recovery: {avg_rec:.2f}")
    return partials, num_deg, per100, frac, avg_rec