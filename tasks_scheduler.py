import os

def build_tasks(config):
    tasks = []
    exp = config["experiment"]
    paths = config["paths"]

    for seed in exp["seeds"]:
        for dimension in exp["dims"]:
            for n_class in exp["n_classes"]:
                for n_cluster in exp["n_clusters_per_class"]:
                    for ratio in exp["ratio_affected"]:
                        for drift_type in exp["drift_types"]:
                            for approach_id in exp["approaches"]:

                                output_folder = os.path.join(
                                    paths["results_root"],
                                    f"seed{seed}",
                                    f"dim{dimension}",
                                    f"class{n_class}",
                                    f"cluster{n_cluster}",
                                    f"ratio{ratio}",
                                    f"scenario_{drift_type}",
                                    f"approach{approach_id}"
                                )
                                os.makedirs(output_folder, exist_ok=True)

                                tasks.append((
                                    n_class,
                                    n_cluster,
                                    ratio,
                                    seed,
                                    dimension,
                                    drift_type,
                                    output_folder,
                                    approach_id,
                                    exp["window_size"]
                                ))
    return tasks
