import numpy as np
from dlscanner.samplers.ML import ML
from dlscanner.gen_scanner import sampler


def obs(x):
    F = (2 + np.cos(x[:, 0]/7)*np.cos(x[:, 1]/7)*np.cos(x[:, 2]/7))**5
    return np.array(F)


def likelihood(exp_value, std, th):
    ll = np.exp(- (exp_value - th)**2/(2*std**2))
    return ll


def true_class(x, exp_v=100, stdv=5):
    th = obs(x)
    ll = likelihood(exp_v, stdv, th)
    for q, item in enumerate(ll):
        if (item > 0.5):
            ll[q] = 1
        else:
            ll[q] = 0
    return np.array(ll)



ndim = 3
limits = [[-10*np.pi, 10*np.pi]]*ndim
num_FC_layers = 3
neurons = 100
model = ML.MLP_Classifier(ndim, num_FC_layers, neurons)
optimizer = "adam"
loss = 'binary_crossentropy'

mysam = sampler(
    true_class, ndim, limits=limits, method='ML', model=model,
    optimizer=optimizer, loss=loss
)



xtst = mysam.genrand(1000)
tctst = true_class(xtst)
pctst = np.round(model.predict(xtst)).flatten()
print((tctst == pctst).sum())

mysam.advance(verbose=0)

xtst = mysam.genrand(1000)
tctst = true_class(xtst)
pctst = np.round(model.predict(xtst)).flatten()
print((tctst == pctst).sum())

