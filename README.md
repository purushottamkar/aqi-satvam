# Robust statistical calibration and characterization of portable low-cost air quality monitoring sensors to quantify real-time O<sub>3</sub> and NO<sub>2</sub> concentrations in diverse environments
This repository presents an implementation of the distance weighted k-nearest neighbors algorithm with a learnt metric (KNN-D(ML) for short) for the purpose of calibrating Alphasense sensors measuring atmospheric O<sub>3</sub> and NO<sub>2</sub> concentrations. The repository also presents some comparitive and ablation studies. The accompanying paper can be found at [[1]][ref1]

## Setup

### Installing Python Library Dependencies

Our implementation of the KNN-D(ML) makes use of various standard libraries. A minimal list is provided in the [requirements.txt](requirements.txt) file. If you are a pip user, please use the following command to install these libraries:

```sh
pip3 install -r requirements.txt
```

For conda users:

```sh
conda install requirements.txt
```

**Note about version dependency**: although the requirements file specifies version dependencies to be exact, this is to err on the side of caution. For most modules (e.g. `numpy` or `scikit-learn`), a more recent version should work too and we are not aware of any version specific dependency on the functioning of KNN-D(ML). If you wish to avoid version-related complications, you may try using the following command instead. This does not impose a version requirement on any module and will either use the version you currently have, or else download the latest one from the pip repository in case you dont have a certain module.

```sh
pip3 install -r requirements_noversion.txt
```

## Reproducing results in the accompanying paper

The repository contains python scripts implementing various calibration algorithms and supporting code. However, a Jupyter notebook [experiments.ipynb](experiments.ipynb) implements an easily accessible wrapper that can be used to reproduce some of the results reported in the accompanying paper [[1]][ref1] on two datasets that are also included in this repository. Please refer to the last section of the notebook **Final Results**

**Note**: please note the following while attempting to reproduce results of the accompanying paper
1. While 12 datasets were used to produce results in the paper, only 2 of those, namely DD1(Jun) and DD1(Oct), are provided as a part of this repository in the directory [data/](data/). Only results relying only on these two datasets can be reproduced.

1. The calibration algorithms rely on train-test splits to obtain an unbiased estimate of the performance of various calibration algorithms. If you wish to reproduce the results in the paper, please use the train-test splits supplied for the 2 datasets DD1(Jun) and DD1(Oct) in the directory [perm/](perm/). The routines included in the repository also allow fresh train-test splits to be created. However, minor differences are to be expected in the results if fresh splits are used to conduct the experiments.    

1. The ranking and win-matrix results included in the notebook do not reproduce those in the accompanying paper (e.g. Table 3 and Table 5). This is because results in the paper are averaged over 12 datasets whereas those in the notebook use only 2 of those datasets, namely DD1(Jun) and DD1(Oct) (which are provided as a part of this repository).

## Contributing
This repository is released under the MIT license. If you would like to submit a bugfix or an enhancement to this repository, please open an issue on this GitHub repository. We welcome other suggestions and comments too (please send an email to purushot@cse.iitk.ac.in)

## License
This repository is licensed under the MIT license - please see the [LICENSE](LICENSE) file for details.

## References
[1]  Ravi Sahu, Ayush Nagal, Kuldeep Kumar Dixit, Harshavardhan Unnibhavi, Srikanth Mantravadi, Srijith Nair, Yogesh Simmhan, Brijesh Mishra, Rajesh Zele, Ronak Sutaria, Vidyanand Motiram Motghare, Purushottam Kar, and Sachchida Nand Tripathi. Robust statistical calibration and characterization of portable low-cost air quality monitoring sensors to quantify real-time O3 and NO2 concentrations in diverse environments. Atmospheric Measurement Techniques, 14(1), 37–52.  (available at [[this link]][ref1]).

[ref1]: https://doi.org/10.5194/amt-14-37-2021
