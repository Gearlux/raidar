"""Every registering module must declare a ``confluid.configurables`` entry point.

``import raidar`` registers NOTHING — every backend-owned name is lazy, so a machine without
torch never imports one — which means discovery runs entirely off the entry-point table: a
scanner imports each module named there, and *the import is what executes the ``@configurable``
decorators*. A module missing from the table registers nothing.

**The failure is silent in every way a developer would notice**, which is why this exists: a
dotted path in YAML imports the module itself so configs keep working, and ``cli.py`` imports its
own backend so runs keep working. Only navigaitor's slot pickers, the form-specs and the
StreamStudio palette go blind — and they fail by offering FEWER options, which reads as a short
menu rather than a bug.

The expectation is DERIVED from the tree with ``ast``, both directions. Static on purpose: no venv
carries every backend's framework, so an import-based check would silently reduce to whichever
extras happen to be installed — exactly the machine where a copy-pasted omission lands. It reads
``pyproject.toml`` rather than installed metadata, because metadata only refreshes on an editable
reinstall and checking it would fail for anyone mid-edit.
"""

import ast
import tomllib
from pathlib import Path
from typing import Dict, Set, cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "raidar"
GROUP = "confluid.configurables"


def _declared() -> Dict[str, str]:
    """The entry-point table as written in ``pyproject.toml``."""
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    return cast(Dict[str, str], data["project"]["entry-points"][GROUP])


def _registers_anything(path: Path) -> bool:
    """True when importing this module would execute a registration.

    Two forms: the ``@configurable`` DECORATOR on a class this project owns, and a bare
    ``configurable(...)`` / ``register(...)`` CALL — which is how ``torchvision_detection.py`` tags the
    torchvision builder functions it cannot decorate.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if getattr(target, "id", getattr(target, "attr", "")) == "configurable":
                    return True
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in {"configurable", "register"} and not isinstance(node.func, ast.Attribute):
                return True
    return False


def _registering_modules() -> Set[str]:
    """Every module in the package whose import registers something, as a dotted path."""
    out = set()
    for path in sorted(SOURCE_DIR.rglob("*.py")):
        if path.name.startswith("_") or path.name == "cli.py":
            continue
        if _registers_anything(path):
            out.add(".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts))
    return out


def test_every_registering_module_is_declared() -> None:
    """The direction that actually bites: a new module, forgotten, silently offers nothing."""
    missing = sorted(_registering_modules() - set(_declared().values()))
    assert not missing, (
        f"these modules register configurables but have no {GROUP} entry point: {missing}. "
        "Add one in pyproject.toml, then `uv pip install -e ./raidar --no-deps` so the "
        "metadata refreshes."
    )


def test_every_declared_path_exists_on_disk() -> None:
    """The converse: a typo'd path is indistinguishable from an absent framework.

    The scanner catches ``ImportError`` per module, so a misspelled entry point just makes the
    module vanish from every picker with no error anywhere.
    """
    for name, dotted in _declared().items():
        path = PROJECT_ROOT / Path(*dotted.split(".")).with_suffix(".py")
        package_init = PROJECT_ROOT / Path(*dotted.split(".")) / "__init__.py"
        assert path.exists() or package_init.exists(), f"entry point {name!r} points at {dotted}, which is not on disk"


def test_entries_are_modules_never_module_colon_attr() -> None:
    """``module:attr`` narrows discovery to ONE object, dropping everything else that module
    registers — and it looks correct."""
    for name, dotted in _declared().items():
        assert ":" not in dotted, f"entry point {name!r} names an attribute ({dotted}); it must name a MODULE"


def test_the_registering_scan_is_not_vacuous() -> None:
    """Guard: if the AST scan matched nothing, both checks above would pass for free."""
    found = _registering_modules()
    assert found, "the @configurable scan found no registering modules — has the scan broken?"
    assert "raidar.lightning_detector" in found
    assert "raidar.rfdetr_detector" in found
    assert "raidar.torchvision_detection" in found
