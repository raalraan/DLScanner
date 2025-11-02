# Explaining Data Anomalies over the NMSSM Parameter Space with Deep Learning Techniques

This is the code for the scan used in the paper
[Explaining Data Anomalies over the NMSSM Parameter Space with Deep Learning Techniques](https://arxiv.org/abs/2508.13912).
It is based on [`NMSSMTools`](https://www.lupm.in2p3.fr/users/nmssm/) and [`MadGraph`](http://madgraph.phys.ucl.ac.be/)
with the help of a custom `spectrum2paramcard.py` based on Johannes Rosskopp original script.

To run the scan it is only necessary to run `python3 run_scan.py`.
In its current state it only works in Linux.

* `run_scan.py`: main script that takes care of setting up and running the scan.
* `my_setup.py`: file that takes care of configurations like working directories,
  size of the different sets of points used in the scan, ranges of constraints,
  parameters used in the scan and their ranges,
  and the numbers that will be parsed from `NMSSMTools` output files.
* `my_classifier.py`: file with the setup of the classifier using the ranges in `my_setup.py`.
  It also contains custom modifications to the `advance` method to use the classifier to keep minimizing the penalty function.
* `pyNMSSMTools.py`: adaptation of `NMSSMTools` to run on multiple CPUs using the configuration from `my_setup.py`.
  it also includes commands to run `MadGraph` when necessary using the commands of `mg_create_output` and `me_calculate_xsec`.
* `spectrum2paramcard.py`: Custom adaptation of `spectrum2paramcard` script by Johannes Rosskopp
  to make it work as an importable module.
* `inp_defs.dat`: Initial input file setting some defaults and used by `pyNMSSMTools.py` to fill in parameter values
  during the scan.
* `LZ_2024+2022.csv`: data for the limits from the LZ collaboration from [Dark Matter Search Results from 4.2 Tonne-Years of Exposure of the LUX-ZEPLIN (LZ) Experiment](https://arxiv.org/abs/2410.17036).
* `Makefile`: file used internally by `pyNMSSMTools.py` for setting up working directories
  using configurations from `my_setup.py`.
* `example_input_output/`: contains examples of input files for NMSSMTools and full outputs for all the numerical studies applied.
  These examples correspond to minimum $\chi^2$, and the mono-H and mono-Z benchmark points.
