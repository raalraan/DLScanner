# Only if using environment variables like SLURM_JOB_ID
from os import environ, urandom
from binascii import hexlify

# Set the random seed using slurm job ID.  If not using slurm, set by hand but
# leave slurm_job_id variable empty
try:
    slurm_job_id = environ["SLURM_JOB_ID"]
    random_seed = int(slurm_job_id)
except KeyError:
    slurm_job_id = ''
    # Use random number for random seed
    random_seed = int(hexlify(urandom(2)), 16)

# Number of workers (or CPUs, tasks, etc) to use in parallel
nworkers = 20
# Number of maximum steps that will be run to improve the neural network and
# collect points
max_steps = 180
# Stop when this number of points has been collected in the target region.
# The scan will terminate either at 'max_steps' or when 'target_points' are
# collected
target_points = 10000

K = 10000
# Require this number of points per class for training.  This means that if
# points that follow all the constraints are not enough to fill min_train,
# then the training class will use min_train points with the smallest penalty
min_train = int(K/2)
L = int(2e7)
# Classifier network setup
hidden_layers = 2
neurons = 1000
epochs = 1000
# VEGAS
use_vegas_map = True
vegas_frac = 3/4

# Auxiliar, base_prefix is not directly imported elsewhere but used below
base_prefix = "/dev/shm/myNMSSMTools" + slurm_job_id

# Working directories will have work_prefix + some index
work_prefix = base_prefix + "/workdir_"
# Directory for input/output files (inp*, spectr*, omega*)
inout_prefix = base_prefix + "/inout/"
# Outputs will be saved but separated according to step where they were found:
# if save_prefix = "./scan_outputs/run_"
# Saved files will be in directories
# ./scan_outputs/run_step_1/
# ./scan_outputs/run_step_2/
# ./scan_outputs/run_step_3/
# ...
save_prefix = "./scan_outputs/run_"

parameter_order = [
    "TANB",
    "LAMBDA",
    "KAPPA",
    "ALAMBDA",
    "AKAPPA",
    "MUEFF",
    "M1",
    "M2",
    "M3",
    "AU3",
    "MQ3",
    "MU3",
]

# Taken from Table 1 of arXiv:2309.07838
# parameter_bounds = [
#     [1.97, 2.58],
#     [0.610, 0.687],
#     [0.307, 0.391],
#     [400, 480],
#     [-621, -402],
#     [238, 291],
#     [255, 3000],
#     [338, 2800],
#     [423, 3000],
#     [-2222, 1288],
#     [825, 3000],
#     [857, 3000],
# ]
# Join the ranges of Table 1 of arXiv:2404.19338 and ranges above
# parameter_bounds = [
#     [1.97, 10.9],
#     [0.013, 0.687],
#     [0.0058, 0.391],
#     [-5000, 480],
#     [-621, 362],
#     [-244, 291],
#     [178, 3000],
#     [304, 5000],
#     [423, 5000],
#     [-5000, 1288],
#     [272, 5000],
#     [570, 5000],
# ]
# Other ranges obtained after some refinements
parameter_bounds = [
    [3.2, 6.2],  # TANB
    [0.07, 0.42],  # LAMBDA
    [0.05, 0.3],  # KAPPA
    [351, 834],  # ALAMBDA
    [-300, -150],  # AKAPPA
    [120, 220],  # MUEFF
    [500, 3000],  # M1
    [750, 10000],  # M2
    [423, 5000],  # M3
    [-5000, 1288],  # AU3
    [272, 10000],  # MQ3
    [570, 10000],  # MU3
]

Ewino_condition = [
    # Range for 2nd lightest neutralino mass
    [140, 270],
    # Range for difference between two lightest neutralino masses
    [5, 30],
]

m125_range = [125.2 - 3, 125.2 + 3]
m95_range = [95.4 - 3, 95.4 + 3]
m650_range = [650 - 25, 650 + 25]
Oh2_upper = 0.12 + 0.001*3

# Spectrum file cites ATLAS (2207.00092 Fig 6) and CMS (2207.00043 Fig 4a)
CU_range = [0.83, 1.14]
CB_range = [0.74, 1.11]
CL_range = [0.82, 1.03]
CV_range = [0.92, 1.00]
CJ_range = [0.83, 1.04]
CG_range = [0.95, 1.14]

# Whether to accept points if only one of the 95 GeV or 650 GeV anomalies is
# found.
# If False then points that have both anomalies are accepted.
# If True then points that have at least one of them are accepted.
accept_95_or_650 = False

# Parsing from the output file
# Tuples with pieces of text that exist in lines that should be parsed.
# By default, the second "word" (index 1 in python) is parsed as the resulting
# value.  If an integer is specified as the last element of the tuple that it
# is used as the index of the "word" that should be parsed as resulting value,
# e.g., if the tuple ends in 2, then the third "word" (index 2 in python) is
# parsed as the resulting value
parse_spectr = [
    (" 35 ", " # lightest neutral scalar"),  # 0
    (" 25 ", " # second neutral scalar"),
    (" 45 ", " # third neutral scalar"),
    (" 36 ", " # lightest pseudoscalar"),
    (" 46 ", " # second pseudoscalar"),
    (" 37 ", " # charged Higgs"),
    (" 1000022 ", " # neutralino(1)"),
    (" 1000023 ", " # neutralino(2)"),
    (" 1000025 ", " # neutralino(3)"),
    (" 1000035 ", " # neutralino(4)"),
    (" 1000045 ", " # neutralino(5)"),  # 10
    (" 1000024 ", " # chargino(1)"),
    (" 1000037 ", " # chargino(2)"),
    (" 1000001", " #  ~d_L"),
    (" 2000001", " #  ~d_R"),
    (" 1000002", " #  ~u_L"),
    (" 2000002", " #  ~u_R"),
    (" 1000003", " #  ~s_L"),
    (" 2000003", " #  ~s_R"),
    (" 1000004", " #  ~c_L"),
    (" 2000004", " #  ~c_R"),  # 20
    (" 1000005", " #  ~b_1"),
    (" 2000005", " #  ~b_2"),
    (" 1000006", " #  ~t_1"),
    (" 2000006", " #  ~t_2"),
    (" 1000011", " #  ~e_L"),
    (" 2000011", " #  ~e_R"),
    (" 1000012", " #  ~nue_L"),
    (" 1000013", " #  ~mu_L"),
    (" 2000013", " #  ~mu_R"),
    (" 1000014", " #  ~numu_L"),  # 30
    (" 1000015", " #  ~tau_1"),
    (" 2000015", " #  ~tau_2"),
    (" 1000016", " #  ~nutau_L"),
    (" 1000021", " #  ~g"),
    (" 10 ", "Omega h^2"),
    (" 20 ", "sigma_p^SI"),
    (" 30 ", "sigma_n^SD"),
    (" 40 ", "sigma_p^SD"),
    (" 24 ", "# MW incl. Delta_MW"),
    ("  6 ", "# Del_a_mu"),  # 40
    (" 61 ", "# Del_a_mu + Theor.Err."),
    (" 62 ", "# Del_a_mu - Theor.Err."),
    ("  1 ", "# BR(b -> s gamma)"),
    (" 11 ", "# (BR(b -> s gamma)+Theor.Err.)"),
    (" 12 ", "# (BR(b -> s gamma)-Theor.Err.)"),
    ("  5 ", "# BR(B+ -> tau+ + nu_tau)"),
    (" 51 ", "# BR(B+ -> tau+ + nu_tau) + Theor.Err."),
    (" 52 ", "# BR(B+ -> tau+ + nu_tau) - Theor.Err."),
    (" 11 ", "# VBF/VH -> H1 -> tautau"),
    (" 12 ", "# ggF -> H1 -> tautau"),  # 50
    (" 13 ", "# VBF/VH -> H1 -> bb"),
    (" 14 ", "# ttH -> H1 -> bb"),
    (" 17 ", "# VBF/VH -> H1 -> gammagamma"),
    (" 18 ", "# ggF -> H1 -> gammagamma"),
    (" 1  1 ", "# N_(1,1)", 2),
    (" 1  2 ", "# N_(1,2)", 2),
    (" 1  3 ", "# N_(1,3)", 2),
    (" 1  4 ", "# N_(1,4)", 2),
    (" 1  5 ", "# N_(1,5)", 2),
    (" 2  1 ", "# N_(2,1)", 2),  # 60
    (" 2  2 ", "# N_(2,2)", 2),
    (" 2  3 ", "# N_(2,3)", 2),
    (" 2  4 ", "# N_(2,4)", 2),
    (" 2  5 ", "# N_(2,5)", 2),
    (" 1  1 ", "# U-type fermions", 2),
    (" 1  2 ", "# D-type fermions", 2),
    (" 1  3 ", "# b-quarks", 2),
    (" 1  4 ", "# taus", 2),
    (" 1  5 ", "# W,Z bosons", 2),
    (" 1  6 ", "# Gluons", 2),  # 70
    (" 1  7 ", "# Photons", 2),
    (" 2  1 ", "# U-type fermions", 2),
    (" 2  2 ", "# D-type fermions", 2),
    (" 2  3 ", "# b-quarks", 2),
    (" 2  4 ", "# taus", 2),
    (" 2  5 ", "# W,Z bosons", 2),
    (" 2  6 ", "# Gluons", 2),
    (" 2  7 ", "# Photons", 2),
    (" 3  1 ", "# U-type fermions", 2),
    (" 3  2 ", "# D-type fermions", 2),  # 80
    (" 3  3 ", "# b-quarks", 2),
    (" 3  4 ", "# taus", 2),
    (" 3  5 ", "# W,Z bosons", 2),
    (" 3  6 ", "# Gluons", 2),
    (" 3  7 ", "# Photons", 2),
    (" 4  1 ", "# U-type fermions", 2),
    (" 4  2 ", "# D-type fermions", 2),
    (" 4  3 ", "# b-quarks", 2),
    (" 4  4 ", "# taus", 2),
    (" 4  5 ", "# W,Z bosons", 2),  # 90
    (" 4  6 ", "# Gluons", 2),
    (" 4  7 ", "# Photons", 2),
    (" 5  1 ", "# U-type fermions", 2),
    (" 5  2 ", "# D-type fermions", 2),
    (" 5  3 ", "# b-quarks", 2),
    (" 5  4 ", "# taus", 2),
    (" 5  5 ", "# W,Z bosons", 2),
    (" 5  6 ", "# Gluons", 2),
    (" 5  7 ", "# Photons", 2),
    ("#    3", "# Muon magn. mom. more than 2 sigma away"),  # 100
    ("#    3", "# Lightest neutralino is not the LSP"),
    ("#    3", "# b -> c tau nu more than 2 sigma away (as SM)"),
    ("#    3", "# Landau Pole below MGUT"),
    ("#    3", "# excluded by ggF/bb->H/A->tautau (ATLAS+CMS)"),
    ("#    3", "# DM direct detection rate too large (SI)"),
    ("#    3", "# DM direct detection rate too large (SD-n)"),
    ("#    3", "# DM direct detection rate too large (SD-p)"),
    ("#    3", "# excluded by trilepton searches for charg(neutral)inos (CMS)"),
    ("#    3", "# excluded by BR(B -> X_s mu +mu-)"),
    ("#    3", "# b -> s gamma more than 2 sigma away"),
    ("#    3", "# b -> d gamma more than 2 sigma away"),
    ("#    3", "# Soft terms > 5 TeV"),  # 112
]

# TODO Is this useful?
#    ("#   ", "Higgsino Xsect*BR into trileptons"),  # 113

parse_omega = [
    (" 1 ", "#  T_f[GeV]"),  # 113
    (" 4 ", "#  omega h^2"),
    (" 3 ", "#  vSigma"),
    (" 0 ", "#  LSP mass"),
    (" 1 ", "#  bino"),
    (" 2 ", "#  wino"),
    (" 3 ", "#  higgsino2"),
    (" 4 ", "#  higgsino1"),  # 120
    (" 5 ", "#  singlino"),
    (" 1 ", "# csPsi [cm^2]"),
    (" 2 ", "# csNsi [cm^2]"),
    (" 3 ", "# csPsd [cm^2]"),
    (" 4 ", "# csNsd [cm^2]"),
    (" 0 ", "#  sigmaV [cm^3/s]"),
    (" 22        22 ",  "# ~o1 ~o1 -> A,A"),
    (" 22        23 ",  "# ~o1 ~o1 -> A,Z"),
    (" 24       -24 ",  "# ~o1 ~o1 -> W+ W-"),
    ("  5        -5 ",  "# ~o1 ~o1 -> b B"),  # 130
    (" 15       -15 ",  "# ~o1 ~o1 -> l L"),
    ("  6        -6 ",  "# ~o1 ~o1 -> t T"),
    (" 25        36 ",  "# ~o1 ~o1 -> h1 ha"),
    (" 23        23 ",  "# ~o1 ~o1 -> Z Z"),
    (" 23        25 ",  "# ~o1 ~o1 -> Z h1"),
    ("#    3", "# DM direct detection rate too large (SI)"),
    ("#    3", "# DM direct detection rate too large (SD-n)"),
    ("#    3", "# DM direct detection rate too large (SD-p)"),  # 138
]
