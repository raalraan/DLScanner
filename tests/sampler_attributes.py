import numpy as np
import matplotlib.pyplot as plt
from DLScanner.gen_scanner import sampler
from user_model import user_function


# Small function for 3D plotting
def scatter3d_plot(data, title=None, alpha=1.0, savefile=None):
    plt.figure()
    ax = plt.axes(projection='3d')
    if title is not None:
        ax.set_title(title)
    ax.scatter3D(*data.T, s=1, alpha=alpha)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% sampler and sampler.advance %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# setup
ndim = 3
limits = [[-10*np.pi, 10*np.pi]]*ndim
hidden_layers = 4
neurons = 100
epochs = 1000
use_vegas_map = True
vegas_frac = 0.5
verbose = 1
K = 10000

# Instantiate sampler and do first training
my_sampler = sampler(
    user_function, ndim, limits=limits, K=K,
    method='Classifier', epochs=epochs,
    verbose=verbose,
    use_vegas_map=use_vegas_map, vegas_frac=0.5
)

steps = 8
my_sampler.advance(steps)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% sampler.sample and sampler.sample_list         %%
# %% sampler.sample_out and sampler.sample_out_list %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
samples = my_sampler.samples
samples_out = my_sampler.samples_out.flatten()
# Discard some burn-in steps from the beginning
n_dis = 3  # Number of initial samples to discard
samples_bi = np.concatenate(
    my_sampler.samples_list[n_dis:]  # Apply some burn-in
)
samples_bi_out = np.concatenate(
    my_sampler.samples_out_list[n_dis:]
).flatten()


scatter3d_plot(
    samples, title="Accumulated samples", alpha=0.2,
    savefile="in_out_samples.png"
)

scatter3d_plot(
    samples[samples_out == 1], title="In-target samples", alpha=0.5,
    savefile="in_samples.png"
)

scatter3d_plot(
    samples_bi[samples_bi_out == 1], title="In-target samples (burn-in)", alpha=0.5,
    savefile="in_samples_bi.png"
)

# %%%%%%%%%%%%%%%%%%%%%%%
# %% sampler.histories %%
# %%%%%%%%%%%%%%%%%%%%%%%
# Show improvement of loss function with iterative trainings
for j in range(len(my_sampler.histories)):
    plt.plot(
        my_sampler.histories[j].history['loss']
    )
plt.yscale('log')
plt.savefig("histories.png")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% sampler.vegas_map_gen %%
# %% sampler.model         %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
veg_sample, _ = my_sampler.vegas_map_gen(K*100)
pred_labels = my_sampler.model(veg_sample).numpy().flatten()
true_labels = user_function(veg_sample)

scatter3d_plot(
    veg_sample, title="VEGAS map samples", alpha=0.1,
    savefile="vegas_map_gen.png"
)

scatter3d_plot(
    veg_sample[pred_labels > 0.5][:K], title=r"DL selected samples", alpha=0.2,
    savefile="vegas_model_pred.png"
)

sam_selK = veg_sample[pred_labels > 0.5][:K]
true_lab_K = user_function(sam_selK)

scatter3d_plot(
    sam_selK[true_lab_K > 0.5], title="In-target collected samples", alpha=0.2,
    savefile="vegas_pred_intarget.png"
)
