"""The ONE place raidar imports TensorFlow — and the ``KERAS_BACKEND`` ordering that must precede it.

The twin of :mod:`raidar._keras`, for the mirror image of that module's problem.
:mod:`recordstream.keras` defaults ``KERAS_BACKEND`` to the first backend actually INSTALLED, in
the order ``torch`` -> ``tensorflow`` -> ``jax``. Right for a Keras run — any engine will do —
and WRONG here: on a machine carrying both ``raidar[lightning]`` and ``raidar[tensorflow]`` (the
workspace's own setup) torch wins, and the TensorFlow backend would then drive a torch-backed
``keras.Model`` with ``tf.GradientTape``, which cannot work.

Keras reads the variable ONCE, at import time, so the default has to be in place before ANY
module imports keras — which is why this module hands over the ``keras`` handle **as well as**
``tf``: importing keras from :mod:`raidar._keras` beside ``tf`` from here would let isort sort
``_keras`` ahead of ``_tensorflow``, importing keras one line before the default that was
supposed to steer it. ``setdefault``, not an assignment: an explicit ``KERAS_BACKEND=jax`` still
wins, and the backend's ``initialize()`` then fails naming the mismatch.
"""

import os

# BEFORE the imports below, and before anything else imports keras — see the module docstring.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf  # noqa: E402 - deliberately after the environment default above
from recordstream.keras import RecordSequence, keras, keras_backend  # noqa: E402

__all__ = ["RecordSequence", "keras", "keras_backend", "tf"]
