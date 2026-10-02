"""The ONE spelling of ``import keras`` in this package — re-exported from recordstream.

Keras 3 reads ``KERAS_BACKEND`` at **import time** and its own default is ``tensorflow``, which
this project does not require (Keras is an API; the compute engine is the operator's choice). So
something has to set the variable to an engine that is actually installed BEFORE keras is first
imported, and that something must sit in the lowest layer — ``recordstream.keras`` — because
import sorters put a third-party import above a first-party one: a shim here would sort BELOW
``from recordstream.keras import RecordSequence`` and lose the race.

This module therefore adds nothing but a name. A raidar module reaches keras through either
``from raidar._keras import keras`` or ``from recordstream.keras import keras`` — the same
object — and never ``import keras`` directly.
"""

from recordstream.keras import keras, keras_backend

__all__ = ["keras", "keras_backend"]
