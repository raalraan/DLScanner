import numpy as np
from .utilities.vegas import vegas_map_samples
from .samplers import ML


class sampler():
    """A generic scanner for a user-defined function

    Args:
        ndim (int): Number of dimensions in the function.
        user_fun (callable): A function that takes a ndim-dimensional vector
            as input and returns at least a single value.  Note that some
            samplers require the output to be a single number per vector
            input.
        limits (array-like of floats): limits for the parameters of the
            function.  It must be an array with ndim rows and two columns.
        outdim (optional, int): Dimension of the output
        args (Optional): Arguments to pass to user_fun besides the ndim vector.
        method (TODO): TODO (This must be used to decide the type of network
            or sampling process to use)
        samples0 (Optional, array-like of floats): an array with an initial set
            of values.  If not given, a random input is generated.
        out0 (Optional, array-like of floats): an array containing the output
            of the function using samples0.  If not given, an initial output is
            obtained from a random  input.
        seed (Optional, int or sequence of int): Seed for the random number
            generator.

    """

    def __init__(
        self,
        user_fun,
        ndim,
        limits,
        outdim=1,
        method="Classifier",
        model=None,
        optimizer="Adam",
        loss=None,
        samples0=None,
        out0=None,
        K=100,
        randpts=None,
        L=None,
        neurons=100,  # Should start as None?
        hlayers=4,  # Should start as None?
        learning_rate=0.001,  # Should start as None?
        epochs=1000,  # Should start as None?
        batch_size=32,  # Should start as None?
        verbose=1,
        args=None,  # TODO Not used for the time being
        use_vegas_map=True,
        vegas_frac=None,
        seed=42,
        callbacks=None,
    ):
        self.samples0 = samples0
        self.out0 = out0
        self.ll = user_fun
        self.method = method
        if method == "Classifier":
            self.model = ML.MLP_Classifier(ndim, hlayers, neurons)
            if loss is None:
                self.loss = "binary_crossentropy"
            else:
                self.loss = loss
        elif method == "Regressor":
            self.model = ML.MLP_Regressor(ndim, hlayers, neurons)
            if loss is None:
                self.loss = "mean_absolute_error"
            else:
                self.loss = loss
        elif method == "Custom":
            if model is None:
                raise ValueError(
                    "A `model` parameter is needed for the 'Custom' method. "
                    "See DLScanner.samplers.ML.ML for available models "
                    "or build a custom network."
                )
            else:
                self.model = model

            if loss is None:
                raise ValueError(
                    "A `loss` parameter is needed for the 'Custom' method. "
                )
            else:
                self.loss = loss

        self.optimizer = optimizer
        self.ndim = ndim
        self.outdim = outdim
        self.limits = limits
        self.limits_arr = np.array(limits)
        self.K = K
        if randpts is None:
            self.randpts = int(K/10 + 0.5)
        else:
            self.randpts = randpts
        if L is None:
            self.L = 100*K
        else:
            self.L = L
        self.neurons = neurons
        self.hlayers = hlayers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # ==================================================
        self.verbose = verbose
        if verbose > 1:
            self.TFverbose = 1
        else:
            self.TFverbose = 0
        self.inited = False
        self.samples = np.empty((0, ndim))
        self.samples_list = []
        self.samples_out = np.empty((0, self.outdim))
        self.samples_out_list = []
        self.use_vegas_map = use_vegas_map
        if use_vegas_map and vegas_frac is None:
            self.vegas_frac = 0.1
        else:
            self.vegas_frac = vegas_frac
        self.vegas_map_gen = None
        self.callbacks = callbacks

        self.rng = np.random.default_rng(seed)
        self.histories = []

        self.initmodel(epochs, batch_size, verbose)

        if len(list(limits)) != ndim:
            print(
                "The value of `ndim` must match the number of pairs in `limits`"
            )

    # TODO This may need to be changed depending on choices of prior --
    def genrand(self, npts):
        dim = self.ndim
        limits_arr = self.limits_arr
        p0 = self.rng.uniform(
            low=limits_arr[:, 0],
            high=limits_arr[:, 1],
            size=[npts, dim]
        )
        return p0

    def initmodel(
        self,
        epochs=None,
        batch_size=None,
        verbose=None,
        callbacks=None,
    ):
        optimizer = self.optimizer
        loss = self.loss
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
            TFverbose = self.TFverbose
        else:
            TFverbose = max(0, verbose - 1)
        npts = self.K
        if callbacks is None:
            callbacks = self.callbacks

        if self.samples0 is None:
            xinit = self.genrand(npts)
            # outinit = np.empty((npts, 1))
            #     outinit[k] = np.array(self.ll(xinit[k]))
            # TODO
            outinit = np.array(self.ll(xinit))
        else:
            xinit = self.samples0
            outinit = self.out0

        outinit_rs = outinit.reshape(npts, self.outdim)

        self.samples = np.append(
            self.samples,
            xinit,
            axis=0
        )
        self.samples_list.append(xinit)
        self.samples_out = np.append(
            self.samples_out,
            outinit_rs,
            axis=0
        )
        self.samples_out_list.append(outinit_rs)

        classvals = np.unique(outinit_rs)
        if len(classvals) == 2:
            fltr0 = (outinit_rs == classvals[0]).flatten()
            fltr1 = (outinit_rs == classvals[1]).flatten()
            nc0 = fltr0.sum()
            nc1 = fltr1.sum()
            xbstc = np.empty((0, self.ndim))
            ybstc = np.empty((0, self.outdim))
            if nc1 < nc0:
                newnc1 = 0
                while newnc1 < nc0:
                    xbstc = np.append(
                        xbstc,
                        xinit[fltr1],
                        axis=0
                    )
                    ybstc = np.append(
                        ybstc,
                        outinit_rs[fltr1],
                        axis=0
                    )
                    newnc1 = xbstc.shape[0]

                xinit_new = np.append(
                    xinit[fltr0],
                    xbstc,
                    axis=0
                )
                outinit_new = np.append(
                    outinit_rs[fltr0],
                    ybstc,
                    axis=0
                )
            elif nc0 < nc1:
                newnc0 = 0
                while newnc0 < nc1:
                    xbstc = np.append(
                        xbstc,
                        xinit[fltr0],
                        axis=0
                    )
                    ybstc = np.append(
                        ybstc,
                        outinit_rs[fltr0],
                        axis=0
                    )
                    newnc1 = xbstc.shape[0]

                xinit_new = np.append(
                    xinit[fltr1],
                    xbstc,
                    axis=0
                )
                outinit_new = np.append(
                    outinit_rs[fltr1],
                    ybstc,
                    axis=0
                )
            else:
                xinit_new = xinit
                outinit_new = outinit_rs

        self.model.compile(optimizer=optimizer, loss=loss)
        self.histories.append(
            self.model.fit(
                xinit_new, outinit_new,  # TODO HERE
                epochs=epochs, batch_size=batch_size, verbose=TFverbose,
                callbacks=callbacks
            ))
        self.inited = True

    def advance(
        self,
        steps=1,
        epochs=None,
        batch_size=None,
        verbose=None,
        callbacks=None,
    ):
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
            TFverbose = self.TFverbose
        else:
            TFverbose = max(0, verbose - 1)
        if callbacks is None:
            callbacks = self.callbacks

        for j in range(steps):
            xsug, llsug = self.suggestpts()
            self.samples = np.append(self.samples, xsug, axis=0)
            self.samples_list.append(xsug)
            self.samples_out = np.append(self.samples_out, llsug, axis=0)
            self.samples_out_list.append(llsug)

            if self.method == "Classifier" and verbose > 0:
                nintgt = (self.samples_out > 0.5).sum()
                print(
                    "Step {}, Number of points in-target: {}".format(
                        j + 1, nintgt
                    )
                )

                xtrain_here, ytrain_here = self.class_equalize()
            else:
                xtrain_here = self.samples
                ytrain_here = self.samples_out

            self.histories.append(
                self.model.fit(
                    xtrain_here, ytrain_here,
                    epochs=epochs, batch_size=batch_size, verbose=TFverbose,
                    callbacks=callbacks
                ))

    def suggestpts(
        self,
        npts=None, randpts=None, L=None, limits=None,
        verbose=None, use_vegas_map=None, vegas_frac=None
    ):
        # randpts: number of points chosen at random that will be
        #     added to the suggested points, randpts < self.K
        # L: number of points that will be used to test the
        #     model in order to obtain suggested points,
        #     L > self.K
        if npts is None:
            npts = self.K
        if randpts is None:
            randpts = self.randpts
        if L is None:
            L = self.L
        if verbose is None:
            verbose = self.verbose
            TFverbose = self.TFverbose
        else:
            TFverbose = max(0, verbose - 1)
        if use_vegas_map is None:
            use_vegas_map = self.use_vegas_map
        if vegas_frac is None:
            vegas_frac = self.vegas_frac
        _randpts = randpts
        # Try to  predict the observable for several points using what the
        # machine learned
        xtry = self.genrand(L)

        if use_vegas_map:
            vegas_pts = int(vegas_frac*L + 0.5)
            # Train a vegas map
            if verbose > 0:
                print("Training vegas map using accumulated samples")
            map_vg = vegas_map_samples(
                self.samples, self.samples_out.flatten(), self.limits
            )
            self.vegas_map_gen = map_vg
            # Discard jacobian
            xtry_vg, _ = map_vg(L)
            xtry = np.concatenate([
                xtry[:L - vegas_pts],
                xtry_vg[:int(vegas_frac*L)]
            ])

        ptry = self.model.predict(
            xtry,
            batch_size=self.batch_size, verbose=TFverbose)

        # FIXME Consider cases other than classification
        xcand = xtry[(ptry > 0.5).flatten()]

        if xcand.shape[0] < npts - randpts:
            _randpts = npts - xcand.shape[0]
            if verbose > 0:
                print(
                    "Tried {} points, {} points survived, filling with random points".format(
                        L, xcand.shape[0]
                    )
                )
        else:
            if verbose > 0:
                print(
                    "Tried {} points, {} points survived, choosing {} points".format(
                        L, xcand.shape[0], npts
                    )
                )

        # Use the points according to the class predicted by the model
        # but pass the correct class. In this way, the points that the
        # model got wrong should be corrected
        xsel = xcand[:npts - _randpts]
        # TODO Check what would be good/interesting stats
        # check1 = (llsel < 100 + 2*10)*(llsel > 100 - 2*10)
        # check2 = ((ptry[fltrcand][:npts - _randpts] - llsel)
        #           / (ptry[fltrcand][:npts - _randpts] + llsel))
        # stats1 = np.sum(check1)/llsel.shape[0]
        # stats2 = np.sum(check2 < 1.e-3)/llsel.shape[0]
        # stats3 = np.sum(check1*(check2 < 1.e-3))/ll1sel.shape[0]

        # Append `_randpts` more points chosen at random
        xnew = self.genrand(_randpts)
        xout = np.append(xsel, xnew, axis=0)
        # llout = np.empty((npts, 1))
        # for k in range(npts):
        #     llout[k] = np.array(self.ll(xout[k]))
        llout = np.array(self.ll(xout))
        return xout, llout.reshape(xout.shape[0], self.outdim)

    def class_equalize(self):
        ccount = [
            (self.samples_out == 0.0).sum(),
            (self.samples_out == 1.0).sum()
        ]

        xsam0 = self.samples[self.samples_out.flatten() == 0.0]
        xsam1 = self.samples[self.samples_out.flatten() == 1.0]
        ysam0 = self.samples_out[self.samples_out.flatten() == 0.0]
        ysam1 = self.samples_out[self.samples_out.flatten() == 1.0]

        xtrain_here = np.empty((0, self.ndim))
        ytrain_here = np.empty((0, self.outdim))

        if ccount[0] != ccount[1]:
            if ccount[1] < ccount[0]:
                while xtrain_here.shape[0] < ccount[0]:
                    xtrain_here = np.append(
                        xtrain_here,
                        xsam1,
                        axis=0
                    )
                    ytrain_here = np.append(
                        ytrain_here,
                        ysam1,
                        axis=0
                    )
                xtrain_here = np.append(
                    xtrain_here[:ccount[0]],
                    xsam0,
                    axis=0
                )
                ytrain_here = np.append(
                    ytrain_here[:ccount[0]],
                    ysam0,
                    axis=0
                )
            elif ccount[0] < ccount[1]:
                while xtrain_here.shape[0] < ccount[1]:
                    xtrain_here = np.append(
                        xtrain_here,
                        xsam0,
                        axis=0
                    )
                    ytrain_here = np.append(
                        ytrain_here,
                        ysam0,
                        axis=0
                    )
                xtrain_here = np.append(
                    xtrain_here[:ccount[1]],
                    xsam1,
                    axis=0
                )
                ytrain_here = np.append(
                    ytrain_here[:ccount[1]],
                    ysam1,
                    axis=0
                )
            else:
                xtrain_here = self.samples
                ytrain_here = self.samples_out

        return xtrain_here, ytrain_here
