# RGS Code for Numerical Experiments

This repository contains the source code used to reproduce the numerical experiments reported in the paper:

**Random Grover Search**

## Overview

The code implements numerical experiments for random Grover search dynamics. The scripts generate the numerical results and figures used in the manuscript.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── grover_2_random_oracle.py
├── grover_3_random_oracle.py
├── grover_deter_vs_random.py
└── grover_projected.py
```

## Files

* `grover_2_random_oracle.py`: numerical experiments for the two-oracle randomized Grover setting.
* `grover_3_random_oracle.py`: numerical experiments for the three-oracle randomized Grover setting.
* `grover_deter_vs_random.py`: comparison between deterministic and randomized oracle sequences.
* `grover_projected.py`: generation of projected trajectory figures.

## Software Requirements

The code was tested with Python 3.9.6. The required Python packages are:

```text
numpy
matplotlib
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Reproducing the Numerical Experiments

To reproduce the two-oracle randomized Grover experiments, run:

```bash
python grover_2_random_oracle.py
```

To reproduce the three-oracle randomized Grover experiments, run:

```bash
python grover_3_random_oracle.py
```

To compare deterministic and randomized oracle sequences, run:

```bash
python grover_deter_vs_random.py
```

To generate the projected trajectory figure, run:

```bash
python grover_projected.py
```

The generated figures will be saved in the current directory.

## Parameters and Random Seeds

The main parameters, including the system size, marked-set sizes, intersection size, error parameter, number of trials, and random seeds, are specified directly in the `if __name__ == "__main__":` block of each script.

## Code Availability

The version of the code corresponding to the manuscript will be archived on Zenodo with a permanent DOI after the manuscript version is finalized.
The version corresponding to the manuscript is archived on Zenodo with the DOI:
https://doi.org/10.5281/zenodo.20603404

## Citation

This repository contains the source code for the numerical experiments in our manuscript.  
A formal paper citation will be added once the manuscript is publicly available.

## License

This repository is released under the MIT License.
