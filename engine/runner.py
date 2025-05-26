import pandas as pd
import os
import time
from approaches.base import get_approach
from utils.utils       import format_dataframe
from utils.results_manager import ResultsManager
from approaches.HoeffdingTree import HoeffdingTree
from approaches.HAT import HoeffdingAdaptiveTree
from approaches.PHAT_M import ProactiveHAT_M
from approaches.EFDT import EFDT
from approaches.EFDT_M import EFDT_M
from approaches.PHT_M import ProactiveHT_M
from approaches.PHT_MR import ProactiveHT_MR
from approaches.PHT_S import ProactiveHT_S
from approaches.PHT_MC import ProactiveHT_MC


def engine(n_class, n_cluster_per_class, ratio_affected, seed, dimension, drift_type, output_folder, approach=0, window_size=1000, results_file = "summary.csv", plot_enabled=False):

    input_file = os.path.join(
        "datasets",
        f"seed{seed}",
        f"dim{dimension}",
        f"class{n_class}",
        f"cluster{n_cluster_per_class}",
        f"ratio{ratio_affected}",
        f"scenario_{drift_type}",
        "data.csv"
    )
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"[engine] Input file not found: {input_file}")
    raw_df = pd.read_csv(input_file, header=None)
    
    X, y, stream = format_dataframe(raw_df)
    
    decision_boundaries = []
    list_models = []

    model = get_approach(approach, window_size)

    
    start_time = time.time()
    decision_boundaries, list_models, predictions = model.train(X, y)
    train_time = time.time() - start_time


    params = {
        "seed": seed,
        "dimension": dimension,
        "n_class": n_class,
        "n_cluster_per_class": n_cluster_per_class,
        "ratio_affected": ratio_affected,
        "scenario_type": drift_type,
        "approach": approach,
        "window_size": window_size,
        "train_time": train_time,
        "stream": stream
    }

    mgr = ResultsManager(results_root=output_folder, results_file = results_file, plot_enabled=plot_enabled)
    mgr.save_all(predictions=predictions, models=list_models, decision_boundaries=decision_boundaries, params=params)
    print(f"[Engine] Results saved to {output_folder}")

    


