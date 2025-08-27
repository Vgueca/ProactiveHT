import os
from pympler import asizeof
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from .csv_utils import update_predictions_csv, update_memory_csv
from .metrics import analyze_degradations, compute_prequential_accuracy
from .global_results import store_global_result
from .plot import plot_frames

class ResultsManager:
    def __init__(self, results_root, results_file = "summary.csv", plot_enabled = False, experiment_root = None):
        self.results_root = results_root
        if experiment_root:
            self.summary_csv = os.path.join(experiment_root, results_file)
        else:
            self.summary_csv = os.path.join(results_root, results_file)
        self.plot_enabled = plot_enabled

    def save_all(self, predictions, models, decision_boundaries, params):

        task_dir =self.results_root
        os.makedirs(task_dir, exist_ok=True)

        preds_csv = os.path.join(task_dir, "predictions.csv")
        mem_csv   = os.path.join(task_dir, "memory_usage.csv")

        valid = [(t, p) for t, p in predictions if p is not None]
        labels = [t for t, p in valid]
        preds  = [p for t, p in valid]


        update_predictions_csv(labels, preds, params['approach'], preds_csv)


        instances = [i * params['window_size'] for i in range(len(models))]
        usages    = [asizeof.asizeof(m) / 1024 for m in models]
        update_memory_csv(instances, usages, params['approach'], mem_csv)


        partials, nd, per100, frac, avg = analyze_degradations(valid)
        preq_series = compute_prequential_accuracy(pd.DataFrame({'label': labels, 'prediction': preds}))
        preq_mean = preq_series.mean() #mean accuracy in the different windows 


        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall    = recall_score   (labels, preds, average='macro', zero_division=0)
        f1        = f1_score       (labels, preds, average='macro', zero_division=0)


        # Build record with only available parameters
        record = {
            "scenario_type": params['scenario_type'],
            "approach": params['approach'],
            "dimension": params['dimension'],
            "pre_accuracy_global": float(preq_mean),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "memory_global": usages[-1] if usages else 0.0, #memory of the last model
            "execution_time": float(params.get('train_time', 0.0)),
            "num_degradations": float(nd),
            "degradations_per_100_windows": float(per100),
            "fraction_degraded": float(frac),
            "avg_recovery_time": float(avg)
        }
        
        # optional parameters 
        optional_params = ['seed', 'n_class', 'n_cluster_per_class', 'ratio_affected']
        for param in optional_params:
            if param in params:
                record[param] = params[param]
        
        # key fields
        base_keys = ['scenario_type', 'approach']
        keys = base_keys + [param for param in optional_params if param in record]

        store_global_result(self.summary_csv, record, keys)


        if self.plot_enabled:
            plot_path_parts = ['plots']
            
            if 'seed' in params:
                plot_path_parts.append(f"seed{params['seed']}")
            
            plot_path_parts.append(f"dim{params['dimension']}")
            
            if 'n_class' in params:
                plot_path_parts.append(f"class{params['n_class']}")
            
            if 'n_cluster_per_class' in params:
                plot_path_parts.append(f"cluster{params['n_cluster_per_class']}")
            
            if 'ratio_affected' in params:
                plot_path_parts.append(f"ratio{params['ratio_affected']}")
            
            plot_path_parts.extend([
                f"scenario_{params['scenario_type']}",
                f"approach{params['approach']}"
            ])
            
            plot_folder = os.path.join(self.results_root, *plot_path_parts)
            plot_frames(params.get('stream'), params.get('window_size'), models, decision_boundaries, plot_folder)

        print(f"[ResultsManager] Saved all results for approach {params['approach']}")
