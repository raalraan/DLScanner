from importlib import import_module

tf_not_found = "TensorFlow could not be imported. \
If working in a virtual environment, \
check that the correct one has been activated. \
If you have not installed tensorflow yet \
and want to use GPU, make sure to install the correct version \
for the CUDA library available in your environment.\n\
For more information see:\n\
    https://www.tensorflow.org/install/pip\n\
    https://www.tensorflow.org/install/source#gpu\n\
If you ONLY want to use CPU, use the following command to install it\n\
    python3 -m pip install tensorflow"


def try_tensorflow(import_this=None):
    try:
        if import_this is None:
            imported = import_module('tensorflow')
        else:
            imported = import_module('tensorflow' + '.' + import_this)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(tf_not_found) from None
    return imported
