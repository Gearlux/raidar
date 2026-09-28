"""Every slot the runnable flows at run time must DECLARE itself deferred.

``Partial[T]`` is what tells a deep-flow walker to leave a slot alone until the runnable
flows it with the kwarg that does not exist yet (``params=``, ``dataset=``,
``num_classes=``). Annotating such a slot ``Any`` — or hand-writing
``Union[X, Fluid]``, which is the ``Partial`` alias minus its marker — declares nothing,
and nothing catches it: the class keeps working only because the slot happens to hold a
``PartialClass`` value.

This derives the expectation from the SOURCE rather than restating a list, so a slot
added tomorrow is covered without editing this file.
"""

import importlib.util
import inspect
import re
from typing import Any, List, Set

import pytest
from confluid import partial_param_names

from raidar.huggingface_detector import HuggingFaceDetector
from raidar.lightning_detector import LightningDetector
from raidar.rfdetr_detector import RFDETRDetector
from raidar.torch_detector import TorchDetector

#: fastai is a SELECTABLE extra, so a workspace that did not pick it has no `fastai` at all and
#: `raidar.fastai_detector` cannot be imported. It then drops out of the sweep rather than taking
#: the whole file down — every other runnable's slots stay checked.
RUNNABLES: List[Any] = [LightningDetector, TorchDetector, HuggingFaceDetector, RFDETRDetector]
if importlib.util.find_spec("fastai") is not None:
    from raidar.fastai_detector import FastaiDetector

    RUNNABLES.append(FastaiDetector)


def _slots_flowed_at_runtime(cls: type) -> Set[str]:
    """``self.<name>`` passed to ``flow(...)`` anywhere in the runnable's module."""
    module = inspect.getmodule(cls)
    assert module is not None
    source = inspect.getsource(module)
    return set(re.findall(r"flow\(self\.([a-z_]+)", source))


@pytest.mark.parametrize("cls", RUNNABLES, ids=lambda c: c.__name__)
def test_every_flowed_slot_is_declared_lazy(cls: type) -> None:
    flowed = _slots_flowed_at_runtime(cls)

    undeclared = sorted(flowed - partial_param_names(cls))

    assert not undeclared, (
        f"{cls.__name__}: {undeclared} are flowed at run time but not annotated Partial[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime "
        "kwarg, that raises; for the rest it silently discards the deferral."
    )


def test_the_flow_scan_is_not_vacuous() -> None:
    """The flows live in the shared bases since 2026-08-06; the scan walks the MRO's
    workspace-owned modules, so an empty result still means the dispatch changed."""
    import inspect

    chunks = []
    for klass in LightningDetector.__mro__:
        module = inspect.getmodule(klass)
        if module is not None and module.__name__.split(".")[0] in {"raidar", "matrainer"}:
            chunks.append(inspect.getsource(module))
    flowed = set(re.findall(r"flow\(\s*self\.([a-z_]+)", "\n".join(chunks)))
    assert flowed, "found no flow(self.<slot>) calls anywhere in the MRO — has the dispatch changed?"


def test_the_lazy_scan_sees_this_class_at_all() -> None:
    """Guard: the body scan reads ``__init__`` SOURCE, so a packaged deployment sees none.

    Without this, the assertion above would pass vacuously wherever the scan is blind.
    """
    assert partial_param_names(LightningDetector), "no lazy slots detected — the body scan may be blind here"
