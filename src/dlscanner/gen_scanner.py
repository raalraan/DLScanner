import numpy as np

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
        sample0 (Optional, array-like of floats): an array with an initial set
            of values.  If not given, a random input is generated.
        out0 (Optional, array-like of floats): an array containing the output
            of the function using sample0.  If not given, an initial output is
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
        method="ML",
        model=None,
        optimizer=None,
        loss=None,
        sample0=None,
        out0=None,
        pts_per_step=100,
        randpts=None,
        testpts=None,
        neurons=100,  # Should start as None?
        hlayers=4,  # Should start as None?
        learning_rate=0.001,  # Should start as None?
        epochs=1000,  # Should start as None?
        batch_size=32,  # Should start as None?
        verbose=None,
        args=None,  # TODO Not used for the time being
        seed=42,
    ):
        self.sample0 = sample0
        self.out0 = out0
        self.ll = user_fun
        if method == "ML":
            if model is None:
                raise ValueError(
                    "A `model` parameter is needed for the 'ML' method. "
                    "See dlscanner.samplers.ML.ML for available models."
                )
            else:
                self.model = model
            if optimizer is None:
                raise ValueError(
                    "A `optimizer` parameter is needed for the 'ML' method. "
                    "See dlscanner.samplers.ML.ML for available models."
                )
            else:
                self.optimizer = optimizer
            if loss is None:
                raise ValueError(
                    "A `loss` parameter is needed for the 'ML' method. "
                    "See dlscanner.samplers.ML.ML for available models."
                )
            else:
                self.loss = loss

        self.ndim = ndim
        self.outdim = outdim
        self.limits = limits
        self.limits_arr = np.array(limits)
        self.pts_per_step = pts_per_step
        if randpts is None:
            self.randpts = int(pts_per_step/10 + 0.5)
        else:
            self.randpts = randpts
        if testpts is None:
            self.testpts = 100*pts_per_step
        else:
            self.testpts = testpts
        # TODO Maybe not always using networks, what's next?
        # IDEA: Leave it as None if not using network
        self.neurons = neurons
        self.hlayers = hlayers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # ==================================================
        if verbose is None:
            self.verbose = 1
        else:
            self.verbose = verbose
        self.inited = False
        self.sample = np.empty((0, ndim))
        self.llsample = np.empty((0, self.outdim))

        self.rng = np.random.default_rng(seed)

        if method == "ML":
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
        verbose=None
    ):
        optimizer = self.optimizer
        loss = self.loss
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
        npts = self.pts_per_step

        if self.sample0 is None:
            xinit = self.genrand(npts)
            # outinit = np.empty((npts, 1))
            #     outinit[k] = np.array(self.ll(xinit[k]))
            # TODO
            outinit = np.array(self.ll(xinit))
        else:
            xinit = self.sample0
            outinit = self.out0

        outinit_rs = outinit.reshape(npts, self.outdim)

        self.sample = np.append(
            self.sample,
            xinit,
            axis=0
        )
        self.llsample = np.append(
            self.llsample,
            outinit_rs,
            axis=0
        )

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

        print(
            (outinit_new == 1.0).sum(),
            (outinit_new == 0.0).sum(),
        )

        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(
            xinit_new, outinit_new,  # TODO HERE
            epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.inited = True

    def advance(self, epochs=None, batch_size=None, verbose=None):
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
        xsug, llsug = self.suggestpts()
        self.sample = np.append(self.sample, xsug, axis=0)
        self.llsample = np.append(self.llsample, llsug, axis=0)

        self.model.fit(
            self.sample, self.llsample,
            epochs=epochs, batch_size=batch_size, verbose=verbose)

    # TODO Include vegas map ++
    def suggestpts(
        self,
        npts=None, randpts=None, testpts=None, limits=None,
        verbose=None
    ):
        # randpts: number of points chosen at random that will be
        #     added to the suggested points, randpts < self.pts_per_step
        # testpts: number of points that will be used to test the
        #     model in order to obtain suggested points,
        #     testpts > self.pts_per_step
        if npts is None:
            npts = self.pts_per_step
        if randpts is None:
            randpts = self.randpts
        if testpts is None:
            testpts = self.testpts
        if verbose is None:
            testpts = self.verbose
        _randpts = randpts
        # Try to  predict the observable for several points using what the
        # machine learned
        xtry = self.genrand(testpts)

        ptry = self.model.predict(
            xtry,
            batch_size=self.batch_size, verbose=verbose)

        # FIXME Consider cases other than classification
        xcand = xtry[(ptry > 0.5).flatten()]

        if xcand.shape[0] < npts - randpts:
            _randpts = npts - xcand.shape[0]
            print(
                "Tried {} points, {} points survived, filling with random points".format(
                    testpts, xcand.shape[0], npts
                )
            )
        else:
            print(
                "Tried {} points, {} points survived, choosing {} points".format(
                    testpts, xcand.shape[0], npts
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


