"""Subprocess helper for the transport tests: the real robot-side worker loop serving the synthetic plant."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'robot_side/franka_polymetis'))
from forceharness.real.plant import Plant
from worker import serve

if __name__=='__main__':
    plant=Plant(time.monotonic)
    plant.frozen='--frozen' in sys.argv
    serve(plant,allow_motion='--enable-motion' in sys.argv)
