import numpy as np
from DLScanner.samplers import ML
from DLScanner.gen_scanner import sampler
import matplotlib.pyplot as plt
from user_model import user_function


def print_confusion_matrix(ytrue, ypred):
    print(
        "True 1", (ypred[ytrue == 0] < 0.5).sum(),
        (ypred[ytrue == 0] > 0.5).sum(),
        "\nTrue 0", (ypred[ytrue == 1] > 0.5).sum(),
        (ypred[ytrue == 1] < 0.5).sum(),
    )


ndim = 3
limits = [[-10*np.pi, 10*np.pi]]*ndim
num_FC_layers = 3
neurons = 100
model = ML.MLP_Classifier(ndim, num_FC_layers, neurons)
optimizer = "adam"
loss = 'binary_crossentropy'
verbose = 0
outplot = "gen_scanner_test.png"

naccul = [None]
naccul_vg = [None]
for use_vegas in [True, False]:
    model = ML.MLP_Classifier(ndim, num_FC_layers, neurons)
    # Instantiate sampler and do first training
    mysam = sampler(
        user_function, ndim, limits=limits, method='Classifier', model=model,
        optimizer=optimizer,
        verbose=verbose, epochs=100, use_vegas_map=use_vegas, vegas_frac=0.5
    )

    # Obtain a data set with half 0 and half 1
    frac1 = 0
    cnt0 = 0
    size_tst = 10000
    xtst0 = np.empty((0, ndim))
    xtst1 = np.empty((0, ndim))
    while frac1 < 0.5:
        xtst = mysam.genrand(10000)
        ytst = user_function(xtst)
        if xtst0.shape[0] < 0.5*size_tst:
            xtst0 = np.append(xtst0, xtst[ytst == 0], axis=0)
        if xtst1.shape[0] < 0.5*size_tst:
            xtst1 = np.append(xtst1, xtst[ytst == 1], axis=0)
        cnt0 += (ytst == 1).sum()
        frac1 = cnt0/size_tst

    xtst_halfs = np.append(
        xtst0[:int(size_tst*0.5)],
        xtst1[:int(size_tst*0.5)],
        axis=0
    )
    ytst_halfs = user_function(xtst_halfs)
    # =========================================

    in_cnt = (user_function(mysam.samples) == 1).sum()
    if use_vegas:
        naccul_vg[0] = in_cnt
    else:
        naccul[0] = in_cnt
    pctst = np.round(model.predict(xtst_halfs, verbose=verbose)).flatten()
    print("TRAINING 1")
    print("Confusion matrix:")
    print_confusion_matrix(ytst_halfs, pctst)
    print(
        "Accumulated points in target region:",
        (user_function(mysam.samples) == 1).sum(),
        "of",
        mysam.samples.shape[0]
    )

    for j in range(10):
        mysam.advance()
        pctst = np.round(model.predict(xtst_halfs, verbose=verbose)).flatten()
        print("TRAINING", j + 2)
        print("Confusion matrix:")
        print_confusion_matrix(ytst_halfs, pctst)

        in_cnt = (user_function(mysam.samples) == 1).sum()
        if use_vegas:
            naccul_vg.append(in_cnt)
        else:
            naccul.append(in_cnt)
        print(
            "Accumulated points in target region:",
            in_cnt,
            "of",
            mysam.samples.shape[0]
        )

plt.plot(
    naccul,
    label="No vegas map"
)
plt.plot(
    naccul_vg,
    label="Using vegas map"
)
plt.xlabel("Iterations")
plt.ylabel("Accumulated points in target regions")
plt.legend()
plt.savefig(outplot)

print(
    "Comparison between using and not using vegas map has been saved to",
    outplot
)
