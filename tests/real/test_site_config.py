"""Site configuration (configs/real/robot.example.json), the remote commands built from it, the deploy bundle,
and the robot-side code constraints. No SSH is run."""
import ast
import importlib.util
import io
import json
import tarfile

import pytest

from forceharness.real import transport
from forceharness.real.config import BackendConfig, load_robot_config
from real_helpers import ROBOT_SIDE, ROOT, robot_side

EXAMPLE = ROOT/'configs/real/robot.example.json'


def filled(tmp_path, **changes):
    site = json.loads(EXAMPLE.read_text())
    site['robot_pc'].update(ssh_host='lab-rt', conda_prefix='~/envs/polymetis', python='~/envs/polymetis/bin/python',
                            hardware_config='/opt/fairo/franka_hardware.yaml')
    site['cameras'].update(exterior_serial='1', wrist_serial='2', hand_eye_file=str(ROOT/'configs/real/hand_eye.example.txt'))
    for section, values in changes.items():
        site[section].update(values)
    path = tmp_path/'robot.json'
    path.write_text(json.dumps(site))
    return path


def test_the_example_loads_offline_and_refuses_live():
    robot = load_robot_config(EXAMPLE)
    assert robot.is_example and 'robot_pc.ssh_host' in robot.placeholders() and 'cameras.wrist_serial' in robot.placeholders()
    with pytest.raises(ValueError, match='filled-in site file'):
        robot.require_live()
    assert robot.robot_pc.robot_port == 50051 and robot.robot_pc.gripper_port == 50052
    assert robot.gripper.force_N == 60 and robot.gripper.open_width_m == .08


def test_a_filled_site_file_is_accepted_and_strict(tmp_path):
    robot = load_robot_config(filled(tmp_path)).require_live()
    assert not robot.is_example and robot.robot_pc.ssh_host == 'lab-rt'
    bad = json.loads(filled(tmp_path).read_text())
    bad['robot_pc']['sshhost'] = 'typo'
    (tmp_path/'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Unknown key'):
        load_robot_config(tmp_path/'bad.json')
    with pytest.raises(ValueError, match='Hand-eye file not found'):
        load_robot_config(filled(tmp_path, cameras=dict(hand_eye_file=str(tmp_path/'missing.txt')))).require_live()
    with pytest.raises(ValueError, match='force_N'):
        load_robot_config(filled(tmp_path, gripper=dict(force_N=100.)))


def test_the_site_file_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('FH_ROBOT_CONFIG', str(filled(tmp_path)))
    assert load_robot_config().robot_pc.ssh_host == 'lab-rt'


def test_worker_command_uses_the_site_paths_and_verifies_the_bundle(tmp_path):
    robot = load_robot_config(filled(tmp_path))
    command = transport.worker_command(BackendConfig(motion_enabled=True), robot)
    version, _ = transport.bundle_info()
    assert command.startswith('cd "$HOME"/forceharness_robot_side/'+version+' && sha256sum --status -c SHA256SUMS')
    assert 'export CONDA_PREFIX="$HOME"/envs/polymetis' in command
    assert 'exec "$HOME"/envs/polymetis/bin/python -u worker.py --hardware-config /opt/fairo/franka_hardware.yaml' in command
    assert '--robot-port 50051 --gripper-port 50052 --enable-motion' in command
    assert '--enable-motion' not in transport.worker_command(BackendConfig(), robot)
    assert '--compile-only' in transport.worker_command(BackendConfig(motion_enabled=True), robot, compile_only=True)
    assert transport.ssh_argv(robot, 'x') == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'lab-rt', 'x']
    assert transport.remote_path('/a b') == "'/a b'" and transport.remote_path('~/x y') == '"$HOME"/\'x y\''


def test_the_bundle_covers_every_robot_side_file_and_the_deploy_never_overwrites(tmp_path):
    spec = importlib.util.spec_from_file_location('deploy_robot_side', ROOT/'scripts/real/deploy_robot_side.py')
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    version, hashes = transport.bundle_info()
    assert version.startswith('robot_side_') and set(hashes) == set(transport.FILES)
    assert {p.name for p in ROBOT_SIDE.glob('*.py')} == set(transport.FILES)
    with tarfile.open(fileobj=io.BytesIO(deploy.archive(hashes))) as tar:
        names = tar.getnames()
        sums = tar.extractfile('SHA256SUMS').read().decode()
    assert set(names) == set(transport.FILES) | {'SHA256SUMS'}
    assert all(f'{hashes[n]}  {n}' in sums for n in transport.FILES)
    command = deploy.deploy_command(load_robot_config(filled(tmp_path)), version)
    assert command.startswith('mkdir -p "$HOME"/forceharness_robot_side && if [ -d ')
    assert 'sha256sum -c SHA256SUMS' in command and 'rm ' not in command
    assert deploy.main([]) == 0                                   # prints the bundle, no SSH


def test_robot_side_code_runs_on_python_3_8():
    for path in ROBOT_SIDE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


def test_guard_limits_are_the_paper_values():
    limits = robot_side('guard').Limits()
    assert (limits.speed, limits.angular_speed, limits.lead, limits.contact_lead) == (.008, .15, .008, .025)
    assert (limits.tracking, limits.rotation, limits.displacement, limits.joint_delta) == (.015, .60, .15, .9)
    assert (limits.f_cap, limits.t_cap, limits.stale, limits.lease, limits.duration) == (25., 3., .15, .30, 300.)
    assert (limits.stall_seconds, limits.joint_margin, limits.smin, limits.residual, limits.residual_rise) == (4., .03, .01, 3., 1.)
    for loosen in (dict(f_cap=26.), dict(speed=.009), dict(contact_lead=.03), dict(joint_margin=.02)):
        with pytest.raises(ValueError):
            robot_side('guard').Limits(**loosen)


def test_one_worker_lock_per_deploy_folder():
    assert robot_side('worker').lock_path() == str(ROOT/'robot_side/.worker.lock')
