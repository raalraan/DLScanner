# DLScanner

[Documentation on arXiv](https://arxiv.org/abs/2412.19675)

A scanner package enhanced by Deep Learning (DL) techniques.
This package addresses two significant challenges associated with previously developed DL-based methods: slow convergence in high-dimensional scans and the limited generalization of the DL network when mapping random points to the target space.
To tackle the first issue, we utilize a Similarity Learning (SL) network that maps sampled points into a representation space.
In this space, in-target points are grouped together while out-target points are effectively pushed apart.
This approach enhances the scan convergence by refining the representation of sampled points.
The second challenge is mitigated by training a [VEGAS mapping](https://vegas.readthedocs.io/en/latest/vegas.html#vegas.AdaptiveMap) of the parameter space
to adaptively suggest new points for the DL network.
This mapping is improved as more points are accumulated
and this improvement is reflected in more efficient collection of points
even for relatively small in-target regions.


# Testing

For testing latest commits or making changes
it is recommended to clone this repository
and test any changes locally.

    git clone https://github.com/raalraan/DLScanner.git

Testing works better inside a virtual environment.
The simplest way to create one is by running:

    # Create virtual environment
    python -m venv /path/to/new/virtual/environment
    # Activate virtual environment
    source /path/to/new/virtual/environment/bin/activate

Replace `/path/to/new/virtual/environment` with the path
that you want to contain the files for the virtual environment.
For example, `.venv` in the root of this repository.

Then, install this package for testing by running `pip install -e .` from the
root of this repository.

