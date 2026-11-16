"""Backends: the robot or simulator under the harness.  All of them implement ``backends.base.Backend``.

* ``mujoco``     - robosuite 1.5 / MuJoCo 3.2.7, used by the HiddenPhys benchmark
* ``forcebench`` - adapter that serves ForceHarness as a ForceBench (Isaac Lab) policy
* ``fake``       - a tiny kinematic stand-in used by the offline tests
* the real Franka FR3 backend lives in ``forceharness.real`` (workstation side) and ``robot_side/`` (robot PC side)
"""
from .base import Backend, WrenchSample  # noqa: F401
