[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces

This is the code associated with the paper "[Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces](https://arxiv.org/abs/2109.10964)."

Please cite our work if you find it useful.

    @InProceedings{pmlr-v162-daulton22a,
    title = 	 {Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces},
    author =       {Daulton, Samuel and Eriksson, David and Balandat, Maximilian and Bakshy, Eytan},
    booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
    year = 	 {2022},
    series = 	 {Proceedings of Machine Learning Research},
    publisher =    {PMLR},
    }


## Getting started

From the base `morbo` directory run:

`pip install -e .`

## Structure

The code is structured in three parts.
- The utilities for constructing the acquisition functions and other helper methods are defined in `morbo/`.
- The experiments are found in and ran from within `experiments/`. The `main.py` is used to run the experiments, and the experiment configurations are found in the `config.json` file of each sub-directory.

The individual experiment outputs were left out to avoid inflating the file size.

## Running Experiments

To run a basic benchmark based on the `config.json` file in `experiments/<experiment_name>` using `<algorithm>`:

```
cd experiments
python main.py <experiment_name> <algorithm> <seed>
```

The code refers to the algorithms using the following labels:
```
algorithms = [
    ("morbo", "MORBO"),
]
```

Each folder under `experiments/` corresponds to the experiments in the paper according to the following mapping:
```
experiments = {
    "dtlz2_10d": "DTLZ2 (d=10)",
    "dtlz2_30d": "DTLZ2 (d=30)",
    "dtlz2_100d": "DTLZ2 (d=100)",
    "dtlz3_m2": "DTLZ3 (M=2)",
    "dtlz5_m2": "DTLZ5 (M=2)",
    "dtlz7_m2": "DTLZ7 (M=2)",
    "dtlz3_m4": "DTLZ3 (M=4)",
    "dtlz5_m4": "DTLZ5 (M=4)",
    "dtlz7_m4": "DTLZ7 (M=4)",
    "rover": "Rover",
    "vehicle_safety": "Vehicle Safety",
    "welded_beam": "Welded Beam",
}
```
Note: this code can heavily exploit a GPU if available.

## License
This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
