"""Configuration of the real-robot backend: guard settings (BackendConfig) and the site file (RobotConfig).

BackendConfig holds the limits and geometry the workstation side runs with; its defaults permit observation
only. RobotConfig holds everything that differs between labs: the SSH host of the real-time computer, the
Polymetis environment and ports there, where the robot-side bundle is deployed, the camera serial numbers and
the hand-eye file. It is read from configs/real/robot.json (copy configs/real/robot.example.json and fill it
in; the copy is git-ignored), or from the file named by $FH_ROBOT_CONFIG or --robot-config.
"""
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG = ROOT/'configs/real/robot.json'
ROBOT_CONFIG_EXAMPLE = ROOT/'configs/real/robot.example.json'


@dataclass(frozen=True)
class BackendConfig:
    # Desk TCP of the Franka Hand (flange to fingertip reference, rotated -45 deg about z); the runner
    # replaces the translation with the centre of the finger pads.
    tcp_translation: tuple = (0.0, 0.0, 0.1034)
    tcp_quaternion: tuple = (0.0, 0.0, -0.3826834323650898, 0.9238795325112867)
    force_sign: float | None = None  # Must be calibrated; never copy sim's sign.
    motion_enabled: bool = False
    sample_period_s: float = 0.05   # Polling interval, NOT a guaranteed motion period.
    stale_s: float = 0.5
    sample_timeout_s: float = 2.0
    max_pair_skew_s: float = 0.35
    max_step_m: float = 0.005
    speed_m_s: float = 0.008
    f_cap: float = 25.0
    t_cap: float = 3.0
    settle_s: float = 1.5
    max_orientation_drift_rad: float = 0.05
    bias_position_scope_m: float = 0.005
    bias_rotation_scope_rad: float = 0.03
    bias_width_scope_m: float = 0.003
    gripper_width_check: bool = True    # False: the robot-side worker skips its gripper width checks

    def __post_init__(self):
        if type(self.motion_enabled) is not bool:
            raise ValueError("motion_enabled must be a boolean")
        if type(self.gripper_width_check) is not bool:
            raise ValueError("gripper_width_check must be a boolean")
        if self.force_sign is not None and (type(self.force_sign) is bool or self.force_sign not in (-1, 1)):
            raise ValueError("force_sign must be null, -1 or +1")
        for key, size in (("tcp_translation", 3), ("tcp_quaternion", 4)):
            a = np.asarray(getattr(self, key), dtype=float)
            if a.shape != (size,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {key}")
            object.__setattr__(self, key, tuple(a.tolist()))
        if abs(np.linalg.norm(self.tcp_quaternion) - 1) > 1e-6:
            raise ValueError("tcp_quaternion must be unit length")
        for key, value in asdict(self).items():
            if key not in {"tcp_translation", "tcp_quaternion", "force_sign", "motion_enabled", "gripper_width_check"}:
                if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be finite and positive")
        for key, ceiling in {"speed_m_s": .008, "max_step_m": .005, "f_cap": 25., "t_cap": 3.,
                             "stale_s": .5, "max_pair_skew_s": .35,
                             "max_orientation_drift_rad": .10}.items():
            if getattr(self, key) > ceiling:
                raise ValueError(f"{key} cannot exceed {ceiling}")
        if self.settle_s < 1.5:
            raise ValueError("settle_s cannot be shorter than 1.5 seconds")

    @classmethod
    def load(cls, path: str | Path) -> "BackendConfig":
        # Unknown keys fail instead of silently changing the experiment.
        return cls(**json.loads(Path(path).read_text()))


# ---- site configuration ------------------------------------------------------------------------------
def _placeholder(value) -> bool:
    return isinstance(value, str) and '<' in value and '>' in value


def _section(cls, data, name):
    """Strict section reader: unknown keys fail, keys starting with '_' are notes and are ignored."""
    data = {k: v for k, v in (data or {}).items() if not k.startswith('_')}
    known = {f.name for f in fields(cls)}
    unknown = set(data)-known
    if unknown:
        raise ValueError(f'Unknown key(s) in {name}: {sorted(unknown)}')
    return cls(**data)


@dataclass(frozen=True)
class RobotPCConfig:
    """The real-time computer that runs Polymetis and the robot-side worker (robot_side/franka_polymetis)."""
    ssh_host: str                       # SSH host (an alias in ~/.ssh/config with key login; BatchMode, no prompt)
    conda_prefix: str                   # Polymetis environment; exported as CONDA_PREFIX (Polymetis reads it)
    python: str                         # Python 3.8 of that environment
    hardware_config: str                # Polymetis robot_client franka_hardware.yaml (joint limits)
    deploy_dir: str = '~/forceharness_robot_side'   # content-addressed bundles are unpacked below this folder
    urdf: str | None = None             # Panda URDF for the offline self-test; None = the one shipped with Polymetis
    robot_port: int = 50051             # Polymetis robot server (on the robot PC itself)
    gripper_port: int = 50052           # Polymetis gripper server (on the robot PC itself)
    ssh_options: tuple = ()             # extra ssh options, appended after -o BatchMode=yes -o ConnectTimeout=5

    def __post_init__(self):
        for key in ('ssh_host', 'conda_prefix', 'python', 'hardware_config', 'deploy_dir'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'robot_pc.{key} must be a non-empty string')
        if self.ssh_host.startswith('-'):
            raise ValueError('robot_pc.ssh_host must not start with "-"')
        for key in ('robot_port', 'gripper_port'):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f'robot_pc.{key} must be a TCP port number')
        object.__setattr__(self, 'ssh_options', tuple(str(o) for o in self.ssh_options))


@dataclass(frozen=True)
class GripperConfig:
    """Franka Hand actions through the Polymetis gripper server. Changed for the public release; not yet
    validated on hardware (the paper's runs used a site-specific service with the same 60 N grasp and 80 mm open)."""
    force_N: float = 60.0
    speed_m_s: float = 0.05
    open_width_m: float = 0.08
    command_timeout_s: float = 20.0

    def __post_init__(self):
        if not 0 < self.force_N <= 70:
            raise ValueError('gripper.force_N must be in (0, 70] N (the Franka Hand continuous rating)')
        if not 0 < self.speed_m_s <= .1:
            raise ValueError('gripper.speed_m_s must be in (0, 0.1] m/s')
        if not .078 <= self.open_width_m <= .08:
            raise ValueError('gripper.open_width_m must be in [0.078, 0.08] m (the runner needs >= 78 mm open)')
        if not 5 <= self.command_timeout_s <= 60:
            raise ValueError('gripper.command_timeout_s must be in [5, 60] s')


@dataclass(frozen=True)
class CameraConfig:
    """Two RealSense cameras on the workstation: the wrist D405 (RGB-D, the only one used to measure) and an
    exterior D435 (RGB for the model to look at)."""
    exterior_serial: str
    wrist_serial: str
    hand_eye_file: str = 'configs/real/hand_eye.txt'   # 4x4 T_flange_camera of the wrist camera (relative to the repo)
    python: str | None = None           # interpreter for the capture process; None = the runner's own interpreter
    capture_timeout_s: float = 20.0

    def __post_init__(self):
        for key in ('exterior_serial', 'wrist_serial', 'hand_eye_file'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f'cameras.{key} must be a non-empty string')
        if not 5 <= self.capture_timeout_s <= 120:
            raise ValueError('cameras.capture_timeout_s must be in [5, 120] s')

    @property
    def capture_python(self) -> str:
        return self.python or sys.executable

    @property
    def hand_eye_path(self) -> Path:
        path = Path(os.path.expanduser(self.hand_eye_file))
        return path if path.is_absolute() else ROOT/path


@dataclass(frozen=True)
class RobotConfig:
    robot_pc: RobotPCConfig
    gripper: GripperConfig
    cameras: CameraConfig
    source: str = ''
    is_example: bool = False

    def placeholders(self):
        """Dotted names of values that are still placeholders such as "<robot-pc>"."""
        out = []
        for section in ('robot_pc', 'gripper', 'cameras'):
            for key, value in asdict(getattr(self, section)).items():
                if _placeholder(value):
                    out.append(f'{section}.{key}')
        return out

    def require_live(self):
        """Refuse to drive hardware with the example file or unfilled placeholders."""
        problems = self.placeholders()
        if self.is_example or problems:
            raise ValueError('Live use needs a filled-in site file: copy configs/real/robot.example.json to '
                             'configs/real/robot.json (or point $FH_ROBOT_CONFIG / --robot-config to one) and fill in '
                             + (', '.join(problems) if problems else 'every <...> value'))
        if not self.cameras.hand_eye_path.is_file():
            raise ValueError(f'Hand-eye file not found: {self.cameras.hand_eye_path} '
                             '(calibrate the wrist camera on your mount; configs/real/hand_eye.example.txt shows the format)')
        return self


def load_robot_config(path=None) -> RobotConfig:
    """The site file: path, else $FH_ROBOT_CONFIG, else configs/real/robot.json, else the example (offline only)."""
    if path is None:
        path = os.environ.get('FH_ROBOT_CONFIG') or (ROBOT_CONFIG if ROBOT_CONFIG.is_file() else ROBOT_CONFIG_EXAMPLE)
    path = Path(path)
    data = json.loads(path.read_text())
    unknown = {k for k in data if not k.startswith('_')}-{'robot_pc', 'gripper', 'cameras'}
    if unknown:
        raise ValueError(f'Unknown section(s) in {path}: {sorted(unknown)}')
    return RobotConfig(robot_pc=_section(RobotPCConfig, data.get('robot_pc'), 'robot_pc'),
                       gripper=_section(GripperConfig, data.get('gripper'), 'gripper'),
                       cameras=_section(CameraConfig, data.get('cameras'), 'cameras'),
                       source=str(path), is_example=path.resolve() == ROBOT_CONFIG_EXAMPLE.resolve())
