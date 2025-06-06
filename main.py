import os
import argparse
import yaml
from mpi4py import MPI
import random

from data.scenarios import generate_scenarios
from utils.utils import *
from engine.runner import engine

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate Proactive concept drift experiments using MPI"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Force regeneration of all data scenarios"
    )
    parser.add_argument(
        "--approach",
        type=int,
        help="Run only the specified approach ID (overrides config)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Override config and enable result plotting"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load configuration
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # 2) Apply command-line overrides
    if args.generate:
        config["generate_scenarios"] = True
    if args.approach is not None:
        config["experiment"]["approaches"] = [args.approach]
    if args.plot:
        config["plot_results"] = True

    # 3) Set number of OMP threads
    os.environ["OMP_NUM_THREADS"] = str(config["mpi"]["omp_num_threads"])

    # 4) Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"[Rank {rank}] Using {size} MPI processes.")


    # 5) Only rank 0 generates scenarios
    if rank == 0 and config.get("generate_scenarios", False):
        print("[Rank 0] Generating data scenarios...")
        generate_scenarios(
            config=config
        )
        print("[Rank 0] Data scenarios generation completed.")
    comm.Barrier()

    
    # 6) Build and distribute tasks
    random.seed(42)
    tasks = build_tasks(config)
    random.shuffle(tasks)  
    total = len(tasks)

    for idx in range(rank, total, size):
        (n_class, n_clusters, ratio, seed, dimension,
         drift_type, folder, approach_id,
         window_size) = tasks[idx]

        print(f"[Rank {rank}] Task {idx+1}/{total}: "
              f"seed={seed}, dim={dimension}, drift={drift_type}, approach={approach_id}")
        engine(
            n_class=n_class,
            n_cluster_per_class=n_clusters,
            ratio_affected=ratio,
            seed=seed,
            dimension=dimension,
            drift_type=drift_type,
            output_folder=folder,
            approach=approach_id,
            window_size=window_size,
            results_file=config.get("results_file", "summary.csv"),
            plot_enabled=config.get("plot_results", False)  
        )
        print(f"[Rank {rank}] Completed task {idx+1}/{total}")
    comm.Barrier()

    if rank == 0:
        print("=== ALL EXPERIMENTS COMPLETED ===")

if __name__ == "__main__":
    main()
