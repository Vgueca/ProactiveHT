# Anticipating to Change: A Proactive Approach for Concept Drift Adaptation In Data Streams

## Abstract

Adapting to drifting data streams remains a key challenge in online learning, where effective model adaptation depends on timely concept drift detection. Most existing approaches respond to drift only after distributional changes occur, reacting to concept drift, limiting their ability to prevent the classifier's performance degradation. This work introduces a novel methodology to anticipate concept drift and enable proactive adaptation before the data distribution shift negatively impacts the classifier. We propose four proactive adaptation strategies based on the VFDT algorithm. We evaluate the proposed methods across four scenarios with diverse data stream configurations. Results demonstrate that proactive adaptation reduces the adverse effects of concept drift and improves classification performance. In particular, the proposed strategies consistently outperformed in settings with incremental drift, underscoring the potential of anticipatory approaches and addressing a notable gap in the current literature.

---

## Experimental setup

### 1. Conda environment (`environment.yml`)

In order to reproduce the experimentation conducted in the paper, we strongly recommend to use the file "environment.yml" where all the packages required for running the experimentation are listed.

### 2. Configuration file (`config.yml`)

The configuration file allows to simple organize the different parameters of the experimental setup used to evaluate the proactive adaptation strategies. 

*General settings:*

* `generate_scenarios`: Set to `true` if you want to regenerate all synthetic data streams before running the experiments. If `false`, the script will user the existing datasets (used in the paper)

*Paths:*

* `dataset_root` and `results_root`: Folder where the synthetic datasets and the results, respectively are stored. By default are set to "datasets" and "results". We recommend to do not change this parameters unless necessary.

*MPI and parallelism:*

* `omp_num_threads`: Specifies the number of OpenMPI threads to use per MPI process. Adjust this value based on your machine's available CPU cores. Default is `8`.

*Experiment parameters:*

* `seeds`: List of random seeds used to generate reproducible experiements. By default is `[2,3,4]`.
* `dims`: List of input dimensionalities. In the experimentation streams with 2D, 5D and 10D feature spaces were generated. By default is `[2,5,10]`.
* `n_classes`: Number of classes. The experimentation consist on binary classification task, so it is fixed to `2`.
* `n_cluster_per_class`: Number of cluster per class generated in each stream. By default `[1,2,3,4]`.
* `ratio_affected`: Proportion of the number of cluster affected by the introduced drift. By default `[0.5,1]`.
* `drift_types`: It corresponds to the scenario types described in the paper.
  * Chase: 0
  * Cross: 1
  * Split: 2
  * Merge: 3
* `approaches`: List of approaches IDs to evaluate in the experimentation. The available approaches are:
  * VFDT (HT): 0
  * PHT-M: 2
  * PHT-MR: 3
  * PHT-S: 8
  * PHT-MC: 10
  * HAT: 20
  * EFDT: 21
  * PHAT-M: 22
  * PEFDT-M: 23
* window_size: Size of the sliding window to perform the proactive adaptation. In our experimentation is fixed to `1000` samples

*Outputs*:

* `results_file`: Name of the file that will store the results of all the different experiments for every approach in each stream. Stored in the results path mentioned above.
* `plot_results`: If set to `true`, the script will generate and save visualizations (plot of points + decision boundaries) of each model in each scenario. The output will be different frames that allow to see the change of the points and decision boundaries along the time (just available for 2D streams).

---

## Run experiment

Once set up the experimetal configuration and the environment, we just need to run the "main.py" file specifying the configuration file were we stored the parameters. An example is provided below:

```bash
python main.py --c config.yml
```

Further flags as: --generate (to generate the synthetic streams), --approach INT (to run the experiment just for a certain approach) and --plot (to generate the plots) are provided to configurate the run without necessary change the configuration file. 

### Parallelism 

This project uses MPI via mpi4py library to allow parallel execution. Experiments are automatically distributed across available MPI processes, which significantly speeds up large experimental runs. We strongly recommend to use this option when evaluating more than 2 approaches.

```bash
mpiexec python main.py --c config.yml
```

Note: The number of threads should be specified in the configuration file.

---

## Results

In the results folder we have included the main results obtained throughout the experimentation, and those included in the paper. Specifically, "summary.csv" includes the results for every scenario and the main approaches studied (`VDFT, PHT-M, PHT-MR, PHT-S and PHT-MC`) and "ablation.csv" inlcudes the results for every scenario and the approaches: `HAT, EFDT, PHAT-M and PEFDT-M`. Additionally, for each possible combination of scenario-approach, the corresponding predictions are stored, which can be obtained by navigating trough the folders in results folder: `results/seed2/dim2/class2/cluster1/ratio1/scenario_0/approach0/predictions.csv`



---

If further information is needed contact with us!

 Happy proactive drifting!
