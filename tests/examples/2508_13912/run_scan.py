#!/usr/bin/env python3
import numpy as np
from my_classifier import my_classifier, user_advance
from my_setup import parameter_bounds, max_steps, target_points, random_seed, \
    min_train, K, L, hidden_layers, neurons, epochs, use_vegas_map, vegas_frac
from DLScanner.gen_scanner import sampler

# Some callbacks may improve predictions
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from math import exp


def lr_scheduler(epoch, lr):
    if (epoch % 10) == 0 and epoch != 0:
        return lr*exp(-0.02)
    else:
        return lr


cb_ES = EarlyStopping(
    monitor='loss',
    patience=50,
    min_delta=0.01,
    restore_best_weights=True,
    start_from_epoch=400,
)
cb_LRS = LearningRateScheduler(
    lr_scheduler
)
# =======================================

# Instantiate classifier, see my_classifier.py
cl = my_classifier(min_train)


def classwrap(p_vec):
    return cl.classifier(p_vec, verbose=1)


# setup
ndim = len(parameter_bounds)
limits = parameter_bounds
verbose = 1
batch_size = int(1e5)

# Instantiate sampler and do first training
my_sampler = sampler(
    classwrap, ndim, limits=limits, K=K,
    L=L, batch_size=batch_size,
    method='Classifier', hlayers=hidden_layers, neurons=neurons, epochs=epochs,
    verbose=verbose,
    use_vegas_map=use_vegas_map, vegas_frac=vegas_frac,
    seed=random_seed, threshold_suggest=0.8, callbacks=[cb_ES, cb_LRS],
)

# Initialize termination conditions
dum = 0
got_in_class = np.concatenate(cl.all_classes).sum()

# Custom advance step that tries to go up in penalty function with a classifier
while dum < max_steps and got_in_class < target_points:
    user_advance(my_sampler, cl, min_train, restart_model=True)
    print("STEP", dum + 1, "OF", max_steps, "FINISHED")
    got_in_class = np.concatenate(cl.all_classes).sum()
    dum += 1

# Save points and outputs that gave results to calculate a finite penalty
penalty = np.concatenate(cl.all_penalty)
ok_fltr = penalty < np.inf
np.savetxt("parameters.csv", np.concatenate(cl.inputs)[ok_fltr])
np.savetxt("outputs.csv", np.concatenate(cl.outputs)[ok_fltr])

one_fltr = np.concatenate(cl.all_classes) == 1.0

# Save points and outputs that are in target region, if any
if one_fltr.sum() > 0:
    onein = np.concatenate(cl.inputs)[one_fltr]
    oneout = np.concatenate(cl.outputs)[one_fltr]

    np.savetxt("parameters_in.csv", onein)
    np.savetxt("outputs_in.csv", oneout)
