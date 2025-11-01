import numpy as np
import shutil
from pathlib import Path

from pyNMSSMTools import pyNMSSMTools_MP
from DLScanner.samplers import ML

from my_setup import nworkers, work_prefix, inout_prefix, save_prefix, \
    m125_range, m95_range, m650_range, Ewino_condition, Oh2_upper, \
    accept_95_or_650, CU_range, CB_range, CL_range, CV_range, CJ_range, \
    CG_range

# %%
myNMSSMTools = pyNMSSMTools_MP(
    nworkers, inout_prefix, work_prefix
)

# %%
LZlims = np.loadtxt("LZ_2024+2022.csv", delimiter=",")


def LZlim(dmmass):
    # Factor of 1e36 convert to pb
    return np.interp(dmmass, LZlims[:, 0], LZlims[:, 1]*1e36)


class my_classifier:
    def __init__(
        self,
        ntrain=None,
        penalty_invalid=np.inf,
        save_prefix=save_prefix
    ):
        self.penalty_min = np.inf
        self.results = None
        self.results_penalty = None
        self.results_Dpenalty = None
        self.results_classes = None
        self.results_fclasses = None
        self.conds = []
        self.ntrain = ntrain
        self.fclass_lim = None
        self.inputs = []
        self.outputs = []
        self.all_penalty = []
        self.all_classes = []
        self.all_fclasses = []
        self.all_conds = []
        self.penalty_invalid = penalty_invalid
        self.save_prefix = save_prefix

    def get_res(self, p_vec):
        self.inputs.append(p_vec)
        self.results = myNMSSMTools(p_vec)
        self.outputs.append(self.results)

    # TODO Adapt this to numbers in 'my_setup.py'
    def class_constraints(self):
        res = self.results

        m125mid = np.sum(m125_range)/2
        m125pm = m125_range[1] - m125mid
        m95mid = np.sum(m95_range)/2
        m95pm = m95_range[1] - m95mid
        m650mid = np.sum(m650_range)/2
        m650pm = m650_range[1] - m650mid

        mchi02mid = np.sum(Ewino_condition[0])/2
        mchi02pm = Ewino_condition[0][1] - mchi02mid
        Dm21mid = np.sum(Ewino_condition[1])/2
        Dm21pm = Ewino_condition[1][1] - Dm21mid

        # Get some results into variables
        mh1 = res[:, 0]  # ~95 GeV scalar
        mh2 = res[:, 1]  # SM-like Higgs
        mh3 = res[:, 2]  # ~650 GeV scalar
        Oh2 = res[:, 35]  # Relic density

        # Neutralino masses
        mneut1 = np.abs(res[:, 6])
        mneut2 = np.abs(res[:, 7])
        # mneut3 = np.abs(res[:, 8])

        # muon g - 2
        # Del_a_mu = res[:, 40]
        # Del a_mu range due to theoretical error
        # Del_a_mu_the = res[:, 41:43]

        # Conditions have a format of [expression, string]
        cond_125 = [
            np.abs(mh2 - m125mid) < m125pm,
            "SM-like Higgs around 125.2 GeV"
        ]
        cond_95 = [np.abs(mh1 - m95mid) < m95pm, "~95 GeV Scalar"]
        cond_650 = [np.abs(mh3 - m650mid) < m650pm, "~650 GeV scalar"]

        if accept_95_or_650:
            cond_95_650 = [
                np.logical_or(cond_95[0], cond_650[0]),
                "Either of ~95 GeV or ~650 GeV Scalar"
                + " ({}, {})".format(cond_95[0].sum(), cond_650[0].sum())
            ]
        else:
            cond_95_650 = [
                cond_95[0]*cond_650[0],
                "Got ~95 GeV and ~650 GeV Scalar"
                + " ({}, {})".format(cond_95[0].sum(), cond_650[0].sum())
            ]

        # This is inf when lightest neutralino is LSP, 3 if not
        cond_LSP = [res[:, 101] > 3, "Neutralino is LSP"]

        cond51 = Oh2 < Oh2_upper  # Relic density below +3*sigma
        # cond52 = Oh2*0.9 > 0.12 - 3*0.001  # Relic density above -3*sigma
        # cond_Oh2 = [cond511cond52, "Omega h^2"]
        cond_Oh2 = [cond51, "Omega h^2"]

        # res[105] is 'inf' when SI DD is below exclusion, but also if
        # micromegas failed to run.  Also check Omega h^2
        cond_SIDD = [
            (res[:, 105] == np.inf)*(Oh2 < np.inf),
            "DM direct detection (SI)"
        ]

        # Enforce m_chi02 in range 140-250 GeV
        cond_mchi02 = [
            np.abs(mneut2 - mchi02mid) < mchi02pm,
            str(Ewino_condition[0][0])
            + " GeV < m_chi02 < "
            + str(Ewino_condition[0][1]) + " GeV"
        ]

        # Enforce Delta m(chi02 - m_chi02) in range 5-30 GeV
        Dm21 = mneut2 - mneut1
        cond_Dm21 = [
            np.abs(Dm21 - Dm21mid) < Dm21pm,
            str(Ewino_condition[1][0])
            + " GeV < m_chi02 - m_chi01 < "
            + str(Ewino_condition[1][1]) + " GeV"
        ]

        cond_Ewwindow = [
            cond_mchi02[0]*cond_Dm21[0],
            "Ewino masses window"
        ]

        # Reduced couplings
        h2utype = res[:, 72]
        h2dtype = res[:, 73]
        h2bquark = res[:, 74]
        h2taus = res[:, 75]
        h2WZ = res[:, 76]
        h2gluons = res[:, 77]
        h2photons = res[:, 78]

        cond_CU = (CU_range[0] < h2utype)*(CU_range[1] > h2utype)
        cond_CD = (CB_range[0] < h2dtype)*(CB_range[1] > h2dtype)
        cond_CB = (CB_range[0] < h2bquark)*(CB_range[1] > h2bquark)
        cond_CL = (CL_range[0] < h2taus)*(CL_range[1] > h2taus)
        cond_CV = (CV_range[0] < h2WZ)*(CV_range[1] > h2WZ)
        cond_CJ = (CJ_range[0] < h2gluons)*(CJ_range[1] > h2gluons)
        cond_CG = (CG_range[0] < h2photons)*(CG_range[1] > h2photons)

        cond_redcoup = [
            cond_CU*cond_CD*cond_CB*cond_CL*cond_CV*cond_CJ*cond_CG,
            "Reduced couplings" +
            "(CU: {}, CB: {}, CL: {}, CV: {}, CJ: {}, CG: {})".format(
                cond_CU.sum(),
                (cond_CD*cond_CB).sum(),
                cond_CL.sum(),
                cond_CV.sum(),
                cond_CJ.sum(),
                cond_CG.sum(),
            )
        ]

        # Del_a_mu, 2 sigma, considering theoretical error
        # cond101 = np.abs(Del_a_mu_the - 2.462104095e-09) < 9.82547875e-10
        # cond_a_mu = [np.any(cond101, axis=1), "Delta a_mu"]
        # ============================

        # List of conditions that will be included in final classification
        # Comment one to exclude it from final classification
        cond_ls = [
            cond_125,
            cond_LSP,
            cond_Oh2,
            cond_SIDD,
            # cond_95,
            # cond_650,
            # cond_95_or_650,
            cond_95_650,
            # cond_mchi02,
            # cond_Dm21,
            cond_Ewwindow,
            # cond_a_mu
            cond_redcoup,
        ]

        cond_res = [cond[0] for cond in cond_ls]
        cond_name = [cond[1] for cond in cond_ls]

        self.results_classes = np.prod(cond_res, axis=0).astype(int)
        self.conds = cond_res
        self.all_conds.append(
            cond_res
        )

        self.all_classes.append(self.results_classes)

        if cond_name is not None:
            for k in range(len(self.conds)):
                print(
                    cond_name[k] + ":",
                    "condition met {} out of {}".format(
                        self.conds[k].sum(), self.conds[k].shape[0]
                    )
                )

        print("All conditions met for {} out of {}".format(
            self.results_classes.sum(),
            self.results_classes.shape[0]
        ))

        self.output_saver()

    # Save data files that have passed all constraints
    def output_saver(self):
        save_prefix = self.save_prefix
        save_which = self.results_classes == 1
        step_index = len(self.all_classes)

        if save_which.sum() > 0:
            ind_save = np.arange(save_which.shape[0], dtype=int)[save_which]

            save_dir = save_prefix + "step_" + str(step_index) + "/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            for ind in ind_save:
                shutil.copyfile(
                    inout_prefix + "spectr_" + str(ind) + ".dat",
                    save_dir + "spectr_" + str(ind) + ".dat"
                )
                shutil.copyfile(
                    inout_prefix + "omega_" + str(ind) + ".dat",
                    save_dir + "omega_" + str(ind) + ".dat"
                )

    def penalty_fun(self, verbose=0):
        res = self.results
        penalty_invalid = self.penalty_invalid

        mh1 = res[:, 0]
        mh2 = res[:, 1]
        mh3 = res[:, 2]
        Oh2 = res[:, 35]
        # SIDD = res[:, 117:119].max(axis=1)
        SIDD = res[:, 36]  # Direct detection
        mneut1 = np.abs(res[:, 6])
        mneut2 = np.abs(res[:, 7])
        # mneut3 = np.abs(res[:, 8])

        # muon g - 2
        # Del_a_mu = res[:, 40]
        # Del a_mu range due to theoretical error
        # Del_a_mu_the = res[:, 41:43]

        mdmcothers = np.abs(res[:, 6:35])

        mdmc_min = mdmcothers.min(axis=1)

        # Not a real chi^2 but used to make sure neutralino is the lightest
        # dark matter candidate
        chi2_LSP = ((mneut1 - mdmc_min)/0.001)**2
        chi2_LSP[mneut1 == np.inf] = penalty_invalid
        LSP_pen = chi2_LSP - 1
        LSP_pen[LSP_pen < 0] = 0

        chi2_Oh2 = ((Oh2 - 0.12)/(0.001))**2
        chi2_Oh2[Oh2 < 0.12] = 0.0
        chi2_Oh2[chi2_Oh2 >= penalty_invalid] = penalty_invalid
        Oh2_pen = chi2_Oh2 - 4
        Oh2_pen[Oh2_pen < 0] = 0

        # SIDD_pen = ((Oh2/0.12)*SIDD/LZlim(mneut1))**2
        # Use log10 instead to avoid this being the largest contribution
        SIDD_pen = np.log10((Oh2/0.12)*SIDD/LZlim(mneut1))
        SIDD_pen[Oh2 == np.inf] = penalty_invalid
        SIDD_pen[SIDD_pen < 0] = 0.0

        # Prefer Delta (m_chi02 - m_chi02) in range 5-30 GeV
        Dm21 = mneut2 - mneut1
        # chi2_Dm21 = ((Dm21 - Dm21mid)/Dm21pm)**2
        # chi2_Dm21[mneut1 == np.inf] = penalty_invalid

        # PENALTY INSTEAD OF CHI^2-LIKE FUNCTION
        m125_pen = np.max([
            mh2 - m125_range[1],
            m125_range[0] - mh2
        ], axis=0)
        m125_pen[m125_pen < 0] = 0.0
        m95_pen = np.max([
            mh1 - m95_range[1],
            m95_range[0] - mh1
        ], axis=0)
        m95_pen[m95_pen < 0] = 0.0
        m650_pen = np.max([
            mh3 - m650_range[1],
            m650_range[0] - mh3
        ], axis=0)
        m650_pen[m650_pen < 0] = 0.0

        # Prefer m_chi02 whitin some range
        mchi02_pen = np.max([
            mneut2 - Ewino_condition[0][1],
            Ewino_condition[0][0] - mneut2
        ], axis=0)
        # Prefer Delta (m_chi02 - m_chi02) whitin some range
        Dm21_pen = np.max([
            Dm21 - Ewino_condition[1][1],
            Ewino_condition[1][0] - Dm21
        ], axis=0)
        Ewino_pen = np.max([
            mchi02_pen,
            Dm21_pen
        ], axis=0)
        Ewino_pen[Ewino_pen < 0] = 0.0
        Ewino_pen[mneut2 == np.inf] = penalty_invalid**0.5

        # chi2_a_mu1 = ((Del_a_mu_the - 2.462104095e-09)/9.82547875e-10)**2
        # chi2_a_mu = np.min(chi2_a_mu1, axis=1)
        # amu_pen = chi2_a_mu - 1
        # amu_pen[amu_pen < 0] = 0

        # Reduced couplings
        h2utype = res[:, 72]
        h2dtype = res[:, 73]
        h2bquark = res[:, 74]
        h2taus = res[:, 75]
        h2WZ = res[:, 76]
        h2gluons = res[:, 77]
        h2photons = res[:, 78]

        CU_pen = np.max([
            h2utype - CU_range[1],
            CU_range[0] - h2utype
        ], axis=0)
        CU_pen[CU_pen < 0] = 0.0
        CD1_pen = np.max([
            h2dtype - CB_range[1],
            CB_range[0] - h2dtype
        ], axis=0)
        CD1_pen[CD1_pen < 0] = 0.0
        CB_pen = np.max([
            h2bquark - CB_range[1],
            CB_range[0] - h2bquark
        ], axis=0)
        CB_pen[CB_pen < 0] = 0.0
        CD_pen = np.max([
            CD1_pen,
            CB_pen
        ], axis=0)
        CL_pen = np.max([
            h2taus - CL_range[1],
            CL_range[0] - h2taus
        ], axis=0)
        CL_pen[CL_pen < 0] = 0.0
        CV_pen = np.max([
            h2WZ - CV_range[1],
            CV_range[0] - h2WZ
        ], axis=0)
        CV_pen[CV_pen < 0] = 0.0
        CJ_pen = np.max([
            h2gluons - CJ_range[1],
            CJ_range[0] - h2gluons
        ], axis=0)
        CJ_pen[CJ_pen < 0] = 0.0
        CG_pen = np.max([
            h2photons - CG_range[1],
            CG_range[0] - h2photons
        ], axis=0)
        CG_pen[CG_pen < 0] = 0.0

        # Sigmas as 1/4 of used range
        m125_sig = np.diff(m125_range)/4
        m95_sig = np.diff(m95_range)/4
        m650_sig = np.diff(m650_range)/4

        CU_sig = np.diff(CU_range)/4
        CB_sig = np.diff(CB_range)/4
        CL_sig = np.diff(CL_range)/4
        CV_sig = np.diff(CV_range)/4
        CJ_sig = np.diff(CJ_range)/4
        CG_sig = np.diff(CG_range)/4

        # Add all the reduced coupling penalties
        rcoups_pen = (CU_pen/CU_sig)**2 + (CD_pen/CB_sig)**2**2 \
            + (CL_pen/CL_sig)**2 + (CV_pen/CV_sig)**2 \
            + (CJ_pen/CJ_sig)**2 + (CG_pen/CG_sig)**2

        # List of values included in total penalty function
        # Comment one to exclude it from total
        # Weighting, scaling, or other stuff can be added to penalties when
        # defined or in here
        # Current weights based on average contributions at start of scan
        # TODO Is it possible to set this weights dynamically? They may need
        # to change as scanning evolves due to network+VEGAS map
        fw = [10**4, 1, 1, 1, 10**3, 1, 1, 10**3]
        penalty_ls = [
            fw[0]*(m125_pen/m125_sig)**2,
            fw[1]*Oh2_pen,
            fw[2]*LSP_pen,
            fw[3]*SIDD_pen**2,
            fw[4]*(m95_pen/m95_sig)**2,
            fw[5]*(m650_pen/m650_sig)**2,
            fw[6]*Ewino_pen**2/0.001,
            fw[7]*rcoups_pen,
        ]

        print(
            "Average contribution to penalty from:\n"
            + "125 GeV, Omega h^2, LSP is neutralino, SI dd,\n"
            + "95 GeV, 650 GeV, Ewino window, reduced couplings:"
        )
        for ar in penalty_ls:
            print(
                ar[ar < np.inf].mean()
            )

        penalty = np.sum(penalty_ls, axis=0)
        # If mh3 is not set the calculation most likely failed
        penalty[mh3 == np.inf] = np.inf
        # If Oh2 is not set, then lightest neutralino is not LSP or something
        # failed with micrOMEGAs

        if penalty.min() < self.penalty_min:
            if verbose > 0:
                print("NEW MINIMUM FOUND")
                print("new: {}, previous: {}".format(
                    penalty.min(), self.penalty_min
                ))
            self.penalty_min = penalty.min()

        self.results_penalty = penalty
        self.results_Dpenalty = penalty - self.penalty_min
        self.all_penalty.append(penalty)

    def classifier_old(self, p_vec, verbose=0):
        # fill self.results
        self.get_res(p_vec)
        self.class_constraints()
        ntrain = self.ntrain

        self.penalty_fun(verbose)
        nc0_tot = (np.concatenate(self.all_classes) == 0).sum()
        nc1_tot = np.concatenate(self.all_classes).sum()

        if ntrain is not None and nc1_tot < ntrain:
            if verbose > 0:
                print("Not enough points in class 1 (0: {}, 1: {}). ".format(
                    nc0_tot, nc1_tot
                ))
                print(
                    "I will pick",
                    ntrain,
                    "points with lowest penalty for class 1."
                )
            pen_asrt = self.results_penalty.argsort()
            pen_asrted = self.results_penalty[pen_asrt]
            self.fclass_lim = 0.5*pen_asrted[[ntrain, ntrain - 1]].sum()
            fclasses_pre = np.full((self.results_penalty.shape[0]), False)
            fclasses_pre[pen_asrt[:ntrain]] = True

            self.results_fclasses = fclasses_pre

            return self.results_fclasses.astype(int)
        else:
            return self.results_classes

    def classifier(self, p_vec, verbose=0):
        # fill self.results
        self.get_res(p_vec)
        self.class_constraints()
        ntrain = self.ntrain

        self.penalty_fun(verbose)
        nc0_tot = (np.concatenate(self.all_classes) == 0).sum()
        nc1_tot = np.concatenate(self.all_classes).sum()

        if ntrain is not None and nc1_tot < ntrain:
            if verbose > 0:
                print("Not enough points in class 1 (0: {}, 1: {}). ".format(
                    nc0_tot, nc1_tot
                ))
                print(
                    "I will pick",
                    ntrain,
                    "points with lowest penalty for class 1."
                )

            all_pen = np.concatenate(self.all_penalty)

            all_pen_asrt = all_pen.argsort()
            all_pen_asrtd = all_pen[all_pen_asrt]

            if (all_pen < np.inf).sum() > ntrain:
                self.fclass_lim = 0.5*all_pen_asrtd[[ntrain, ntrain - 1]].sum()
            else:
                print(
                    "Points with finite penalty are less than {}:".format(
                        ntrain
                    ),
                    (all_pen < np.inf).sum()
                )
                self.fclass_lim = np.inf

            for k in range(len(self.all_fclasses)):
                newfclass_k = (
                    self.all_penalty[k] < self.fclass_lim
                ).astype(int)
                self.all_fclasses[k] = newfclass_k

            self.all_fclasses.append(
                (self.results_penalty < self.fclass_lim).astype(int)
            )

            self.results_fclasses = self.all_fclasses[-1]

            return self.results_fclasses
        else:
            return self.results_classes


# %% CREATE A CUSTOM ADVANCE STEP
def user_advance(
    mysam, myclass, ntrain,
    verbose=None, batch_size=None, epochs=None, callbacks=None,
    restart_model=False, full_train=True
):
    xsug, llsug = mysam.suggestpts()
    mysam.samples = np.append(mysam.samples, xsug, axis=0)
    mysam.samples_list.append(xsug)
    mysam.samples_out = np.append(mysam.samples_out, llsug, axis=0)
    mysam.samples_out_list.append(llsug)

    # Custom correction of the classes to keep looking for smaller penalty ====
    if np.concatenate(myclass.all_classes).sum() < ntrain:
        mysam.samples_out = np.concatenate(myclass.all_fclasses).reshape((-1, 1))
        for k in range(len(mysam.samples_out_list)):
            mysam.samples_out_list[k] = myclass.all_fclasses[k].reshape((-1, 1))
    else:
        mysam.samples_out = np.concatenate(myclass.all_classes).reshape((-1, 1))
        for k in range(len(mysam.samples_out_list)):
            mysam.samples_out_list[k] = myclass.all_classes[k].reshape((-1, 1))
    # ========================================================================

    model_retrain(
        mysam, myclass,
        batch_size=batch_size, callbacks=callbacks,
        epochs=epochs, verbose=verbose,
        full=full_train, restart=restart_model
    )


def user_cequalize(mysam, myclass):
    outflat = mysam.samples_out.flatten()
    n0 = (outflat == 0.0).sum()
    n1 = (outflat == 1.0).sum()

    minto = min(n0, n1)

    samples_0 = mysam.samples[outflat == 0.0]
    samples_1 = mysam.samples[outflat == 1.0]
    samouts_0 = mysam.samples_out[outflat == 0.0].reshape(-1, 1)
    samouts_1 = mysam.samples_out[outflat == 1.0].reshape(-1, 1)

    ind0 = np.arange(samples_0.shape[0], dtype=int)
    ind1 = np.arange(samples_1.shape[0], dtype=int)

    np.random.shuffle(ind0)
    np.random.shuffle(ind1)

    sampshuf = np.concatenate(
        [samples_0[ind0][:minto], samples_1[ind1][:minto]]
    )
    outshuf = np.concatenate(
        [samouts_0[:minto], samouts_1[:minto]]
    )

    return sampshuf, outshuf


def model_retrain(
    mysam, myclass,
    batch_size=None, callbacks=None,
    epochs=None, verbose=None,
    full=True, restart=False
):
    if epochs is None:
        epochs = mysam.epochs
    if batch_size is None:
        batch_size = mysam.batch_size
    if verbose is None:
        verbose = mysam.TFverbose
    if callbacks is None:
        callbacks = mysam.callbacks

    neurons = mysam.neurons
    optimizer = mysam.optimizer
    loss = mysam.loss

    if restart:
        mysam.model = ML.MLP_Classifier(mysam.ndim, mysam.hlayers, neurons)
        mysam.model.compile(optimizer=optimizer, loss=loss)

    if full:
        xtrain, ytrain = mysam.class_equalize()
    else:
        xtrain, ytrain = user_cequalize(mysam, myclass)

    if xtrain.shape[0] < batch_size:
        batch_size = xtrain.shape[0]

    mysam.histories.append(mysam.model.fit(
        xtrain, ytrain,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=callbacks
    ))
