import numpy as np
import subprocess as sp
from os import getcwd, chdir, setsid, killpg
import signal
import re
import multiprocessing as mp
from pathlib import Path
from math import ceil, floor
from shutil import rmtree

# Previously just parse_output, but two output files
from my_setup import parse_spectr, parse_omega, parameter_order, \
    Ewino_condition, m125_range, Oh2_upper, m95_range, m650_range, \
    accept_95_or_650, CU_range, CB_range, CL_range, CV_range, CJ_range, \
    CG_range

from spectrum2paramcard import spectrum2paramcard

parse_output = (parse_spectr, parse_omega)

# %% Info for parsing input and output files
p_iden0 = {
    "TANB": "3",
    "LAMBDA": "61",
    "KAPPA": "62",
    "ALAMBDA": "63",
    "AKAPPA": "64",
    "MUEFF": "65",
    "M1": "1",
    "M2": "2",
    "M3": "3",
    "AU3": "11",
    "MQ3": "43",
    "MU3": "46",
}

p_iden1 = {
    "TANB": "# TANB at MZ",
    "LAMBDA": "# LAMBDA",
    "KAPPA": "# KAPPA",
    "ALAMBDA": "# ALAMBDA",
    "AKAPPA": "# AKAPPA",
    "MUEFF": "# MUEFF",
    "M1": "# M1",
    "M2": "# M2",
    "M3": "# M3",
    "AU3": "# AU3",
    "MQ3": "# MQ3",
    "MU3": "# MU3",
}

p_iden_regex = {}
for j in range(len(parameter_order)):
    str0 = str(p_iden0[parameter_order[j]])
    str1 = p_iden1[parameter_order[j]]
    p_iden_regex.update({
        parameter_order[j]: str0 + " .* " + str1
    })

p_order_dict = {}
for j in range(len(parameter_order)):
    p_order_dict.update({parameter_order[j]: j})


# %%

pywd = getcwd()
# Parse the defaults from a default input file
dummyin = pywd + "/inp_defs.dat"
with open(dummyin) as f:
    dmin = f.read()


def replace_pline(p_vec):
    dmin_hr = dmin
    for j in range(len(p_vec)):
        p_label = parameter_order[j]
        p_vec_str = " {:.8E} ".format(p_vec[j])
        p_str_0 = p_iden0[parameter_order[j]]
        p_str_1 = p_iden1[parameter_order[j]]
        dmin_hr = re.sub(
            p_iden_regex[p_label],
            p_str_0 + p_vec_str + p_str_1,
            dmin_hr
        )
    return dmin_hr


def create_inp_file(filename, p_vec):
    with open(filename, "w") as f:
        f.write(replace_pline(p_vec))


def parse_out_file(filenames, parse_out, failed=False):
    nparse0 = len(parse_out[0])
    nparse1 = len(parse_out[1])
    nparse = nparse0 + nparse1  # Maximum number parsed outputs
    dummyvec = np.full((nparse,), np.inf)

    if not failed:
        with open(filenames[0], 'r') as f:
            for line in f:
                for ind, parsing in enumerate(parse_out[0]):
                    if type(parsing[-1]) is int:
                        if all(dum in line for dum in parsing[:-1]):
                            dummyvec[ind] = float(line.split()[parsing[-1]])
                            break
                    else:
                        if all(dum in line for dum in parsing):
                            dummyvec[ind] = float(line.split()[1])
                            break
                if np.sum(dummyvec < np.inf).sum() == len(parse_out[0]):
                    break

        with open(filenames[1], 'r') as f:
            for line in f:
                for ind, parsing in enumerate(parse_out[1]):
                    if type(parsing[-1]) is int:
                        if all(dum in line for dum in parsing[:-1]):
                            dummyvec[ind + nparse0] = float(line.split()[parsing[-1]])
                            break
                    else:
                        if all(dum in line for dum in parsing):
                            dummyvec[ind + nparse0] = float(line.split()[1])
                            break
                if np.sum(dummyvec < np.inf).sum() == len(parse_out[1]):
                    break

    return dummyvec


def pyMG(
    spectrfile, work_prefix="", MadGraph="MG5_aMC_v3_6_2",
    verbose=0
):
    MG_output_dir = work_prefix + '/' + MadGraph + '/output_dir_mg/'

    # TODO Use the correct output file
    spectrum2paramcard(spectrfile, MG_output_dir)

    chdir(MG_output_dir)
    output = sp.check_output(
        ["./bin/madevent", "../me_calculate_xsec"],
        # stdout=stdout_to,
        stderr=sp.DEVNULL,
    )
    chdir(pywd)

    output_ls = str(output).split("\\n")

    # Use np.inf for moments when there is no line with "Cross-section :"
    xsec = np.inf
    for line in output_ls:
        if "Cross-section :" in line:
            # Is cross section always at the forth element from the end?
            xsec = line.split(" ")[-4]
            break

    if verbose > 0:
        for line in output_ls:
            print(line)

    with open(spectrfile, "a") as outfile:
        outfile.write("\n# p p > chi+-_1 chi0_2 [pb]:  " + str(xsec) + "\n")

    rmtree(MG_output_dir + "/Events/run_01", ignore_errors=True)

    return float(xsec)


def pyNMSSMTools(
    p_vec, inout_prefix="", work_prefix="", inout_suffix="",
    NMSSMTools="NMSSMTools_6.1.2", parse_this=parse_output,
    verbose=0, timeout=900
):
    '''
    p_vec: input vector with the parameters to use.
    prefix: part of the input/output files before {inp,spectr,omega}.
        Can include directory.
    suffix: part of the input/output files after {inp,spectr,omega}.
        File extension is not needed.
    NMSSMTools_dir: directory where NMSSMTools 'run' executable is located
    '''
    if verbose == 1:
        stdout_to = sp.DEVNULL
        stderr_to = sp.STDOUT
    elif verbose > 1:
        stdout_to = None
        stderr_to = None
    else:
        stdout_to = sp.DEVNULL
        stderr_to = sp.DEVNULL

    NMSSMTools_dir = work_prefix + "/" + NMSSMTools
    inpfile = inout_prefix + "inp" + inout_suffix + ".dat"
    spectrfile = inout_prefix + "spectr" + inout_suffix + ".dat"
    omegafile = inout_prefix + "omega" + inout_suffix + ".dat"
    outfiles = [spectrfile, omegafile]

    create_inp_file(inpfile, p_vec)

    if getcwd() != NMSSMTools_dir:
        chdir(NMSSMTools_dir)

    failed = False
    # Without timeout
    # sp.run(
    #     ["./run", inpfile],
    #     stdout=stdout_to,
    #     stderr=stderr_to
    # )
    with sp.Popen(
        ["./run", inpfile],
        stdout=stdout_to,
        stderr=stderr_to,
        preexec_fn=setsid
    ) as process:
        try:
            process.communicate(timeout=timeout)
        except sp.TimeoutExpired:
            if verbose > 0:
                print("Timeout reached!")
            killpg(process.pid, signal.SIGINT)
            failed = True

    chdir(pywd)

    res_vec = parse_out_file(outfiles, parse_this, failed)

    # res_vec[7] is the second lightest neutralino mass
    # res_vec[101] is 'inf' when LSP is neutralino
    # res_vec[105] is 'inf' when SI DD is below exclusion
    MG_run_condition = False
    if res_vec[7] < np.inf and res_vec[101] == np.inf:
        # Ewino_window
        mchi02 = np.abs(res_vec[7])
        Dmchi21 = np.abs(res_vec[7]) - np.abs(res_vec[6])
        mh95 = res_vec[0]
        mh125 = res_vec[1]
        mh650 = res_vec[2]
        Oh2 = res_vec[35]

        if accept_95_or_650:
            m95_650_cond = (
                m95_range[0] < mh95 < m95_range[1]
                or m650_range[0] < mh650 < m650_range[1]
            )
        else:
            m95_650_cond = (
                m95_range[0] < mh95 < m95_range[1]
                and m650_range[0] < mh650 < m650_range[1]
            )
        # Reduced couplings
        h2utype = res_vec[72]
        h2dtype = res_vec[73]
        h2bquark = res_vec[74]
        h2taus = res_vec[75]
        h2WZ = res_vec[76]
        h2gluons = res_vec[77]
        h2photons = res_vec[78]

        cond_CU = (CU_range[0] < h2utype)*(CU_range[1] > h2utype)
        cond_CD = (CB_range[0] < h2dtype)*(CB_range[1] > h2dtype)
        cond_CB = (CB_range[0] < h2bquark)*(CB_range[1] > h2bquark)
        cond_CL = (CL_range[0] < h2taus)*(CL_range[1] > h2taus)
        cond_CV = (CV_range[0] < h2WZ)*(CV_range[1] > h2WZ)
        cond_CJ = (CJ_range[0] < h2gluons)*(CJ_range[1] > h2gluons)
        cond_CG = (CG_range[0] < h2photons)*(CG_range[1] > h2photons)

        redcoup_cond = all([
            cond_CU, cond_CD, cond_CB, cond_CL, cond_CV, cond_CJ, cond_CG,
        ])

        if (Ewino_condition[0][0] < mchi02 < Ewino_condition[0][1]) \
                and (Ewino_condition[1][0] < Dmchi21 < Ewino_condition[1][1]) \
                and (m125_range[0] < mh125 < m125_range[1]) \
                and (Oh2 < Oh2_upper) \
                and m95_650_cond \
                and (res_vec[105] == np.inf and Oh2 < np.inf) \
                and redcoup_cond:
            MG_run_condition = True

    xsec = np.inf
    if (not failed) and MG_run_condition:
        print(
            "Running MadGraph because constraints have been satisfied",
        )
        xsec = pyMG(spectrfile, work_prefix=work_prefix)
        print("Got cross section:", xsec, "pb")

    return np.append(res_vec, xsec)


def pyNMSSMTools_array(
    p_array, inout_prefix="", work_prefix="", p_elems=None,
    verbose=0, timeout=600, NMSSMTools="NMSSMTools_6.1.2",
    parse_this=parse_output
):

    if p_elems is None:
        p_elems = list(range(p_array.shape[0]))

    res = []
    for k in p_elems:
        res.append(pyNMSSMTools(
            p_array[k],
            inout_prefix=inout_prefix,
            work_prefix=work_prefix,
            inout_suffix="_" + str(k),
            NMSSMTools=NMSSMTools,
            parse_this=parse_this,
            verbose=verbose,
            timeout=timeout
        ))

    return np.array(res)


def pyNMSSMTools_MP(
    nworkers, inout_prefix="", work_prefix="",
    verbose=0, timeout=600, workdirs_ready=False
):
    '''
    nworkers: number of workers
    base_prefix: the base prefix for where working directories will be created
       and for where the input/output directory will be created.  Input/output
       directory will have a suffix '_inout' appended.
    workdir_prefix: prefix that will be used for working directories.  The
       number of the worker will be appended.
    '''
    if not workdirs_ready:
        create_workdirs(
            work_prefix,
            nworkers,
            verbose
        )
    Path(inout_prefix).mkdir(parents=True, exist_ok=True)

    print(
        "===========================================\n" +
        "\nWork directories have been created as\n    " +
        work_prefix +
        "\nand a worker index.\n\n" +
        "===========================================\n"
    )

    def _pyNMSSMTools_MP(
        p_array,
        inout_prefix=inout_prefix,
        work_prefix=work_prefix,
        verbose=verbose
    ):
        # Create distribution of points
        psh0 = p_array.shape[0]
        pmax = min(nworkers, psh0)

        npmin = floor(psh0/pmax)
        npmax = ceil(psh0/pmax)

        nnpmax = psh0 - npmin*pmax

        npdist = nnpmax*[npmax] + (pmax - nnpmax)*[npmin]

        dist_list = []
        for k in range(pmax):
            dist_list.append(list(
                range(
                    sum(npdist[:k]),
                    sum(npdist[:k + 1])
                )
            ))

        NMSSMTools_inputs = [
            (
                p_array,
                inout_prefix,
                work_prefix + str(j),
                dist_list[j],
                verbose,
                timeout
            )
            for j in range(pmax)
        ]

        with mp.Pool(processes=pmax) as pool:
            r = pool.starmap(pyNMSSMTools_array, NMSSMTools_inputs)

        return np.concatenate(r)

    return _pyNMSSMTools_MP


def create_workdirs(prefix, nworkers=1, verbose=0):
    '''
    nworkers: number of workers
    prefix: for the string at the beginning of working directories.  Used for
        setting the place where NMSSMTools will be installed and executed.
    '''
    if verbose > 0:
        stdoutto = None
        stderrto = None
    else:
        stdoutto = sp.DEVNULL
        stderrto = sp.STDOUT

    print(
        "I will create",
        nworkers,
        "directories and build NMSSMTools in them.",
        "Wait..."
    )

    making = [
        sp.Popen(
            ["make", "workdir=" + prefix + str(j)],
            stdout=stdoutto,
            stderr=stderrto,
        )
        for j in range(nworkers)
    ]

    made = 0
    for makes in making:
        makes.wait()
        made += 1
        print("Finished", made, "of", nworkers)
        if makes.returncode != 0:
            print(
                "Worker", made, "(index {})".format(made - 1),
                "returned", makes.returncode
            )
            print(
                "Try compiling inside worker folder in a terminal"
                " or use verbose larger than 0"
            )
            exit()
