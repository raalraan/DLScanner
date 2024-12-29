import numpy as np


def obs(x):
    F = (2 + np.cos(x[:, 0]/7)*np.cos(x[:, 1]/7)*np.cos(x[:, 2]/7))**5
    return np.array(F)


def likelihood(exp_value, std, th):
    ll = np.exp(- (exp_value - th)**2/(2*std**2))
    return ll


def user_function(x, exp_v=150, stdv=5):
    th = obs(x)
    ll = likelihood(exp_v, stdv, th)
    for q, item in enumerate(ll):
        if (item > 0.5):
            ll[q] = 1
        else:
            ll[q] = 0
    return np.array(ll)


def print_confusion_matrix(ytrue, ypred):
    print(
        "True 1", (ypred[ytrue == 0] < 0.5).sum(),
        (ypred[ytrue == 0] > 0.5).sum(),
        "\nTrue 0", (ypred[ytrue == 1] > 0.5).sum(),
        (ypred[ytrue == 1] < 0.5).sum(),
    )

