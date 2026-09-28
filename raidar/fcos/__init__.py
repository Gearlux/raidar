"""The shared tiny-FCOS spec and its per-framework implementations.

This ``__init__`` imports NOTHING, and that is load-bearing: ``raidar/__init__.py`` resolves
every backend name lazily so a machine without torch never imports it, and a re-export here
would drag a framework in the moment anything under the package — including the
dependency-free ``spec.py`` — is touched. Import the implementation you want by module:
``from raidar.fcos.torch_fcos import TorchFcos``.
"""
