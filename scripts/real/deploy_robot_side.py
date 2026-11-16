#!/usr/bin/env python3
"""Deploy and check the robot-side bundle (robot_side/franka_polymetis) on the real-time computer. Never moves the robot.

The bundle is content-addressed: robot_side_<hash of the file hashes>. It is unpacked into
<robot_pc.deploy_dir>/<bundle> over SSH and every start of the worker verifies its SHA256SUMS; an existing bundle
with the same name is only verified, never overwritten. Settings come from configs/real/robot.json.

    python scripts/real/deploy_robot_side.py                    # print the bundle name and file hashes (no SSH)
    python scripts/real/deploy_robot_side.py --deploy           # copy and verify the bundle on the robot PC
    python scripts/real/deploy_robot_side.py --selftest         # run selftest.py there: guard lifecycle, stock
                                                                # controller class, wrench solve; no robot RPC
    python scripts/real/deploy_robot_side.py --compile-check    # worker --compile-only: reads the robot state and
                                                                # compiles the controller, never sends it
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.config import BackendConfig, load_robot_config  # noqa: E402
from forceharness.real.transport import (FILES, ROBOT_SIDE, bundle_command, bundle_dir, bundle_info,  # noqa: E402
                                         remote_path, ssh_argv, worker_command)


def archive(hashes):
    """Uncompressed tar of the bundle files and their SHA256SUMS, with fixed timestamps."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as tar:
        for name in FILES:
            info = tar.gettarinfo(ROBOT_SIDE/name, arcname=name)
            info.mtime = 0
            with (ROBOT_SIDE/name).open('rb') as source:
                tar.addfile(info, source)
        body = ''.join(hashes[name]+'  '+name+'\n' for name in FILES).encode()
        info = tarfile.TarInfo('SHA256SUMS')
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return data.getvalue()


def deploy_command(robot, version):
    target = remote_path(bundle_dir(robot, version))
    # An existing identical bundle is checked, never overwritten.
    return ('mkdir -p '+remote_path(robot.robot_pc.deploy_dir)+' && '
            'if [ -d '+target+' ]; then cd '+target+' && sha256sum -c SHA256SUMS; '
            'else mkdir '+target+' && tar -xf - -C '+target+' && cd '+target+' && sha256sum -c SHA256SUMS; fi')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deploy', action='store_true', help='copy the bundle to the robot PC and verify it')
    ap.add_argument('--selftest', '--offline-check', dest='selftest', action='store_true',
                    help='run the offline self-test of the deployed bundle on the robot PC')
    ap.add_argument('--compile-check', action='store_true',
                    help='start the worker with --compile-only (reads state, compiles the controller, sends nothing)')
    ap.add_argument('--robot-config', type=Path, default=None,
                    help='site file (default: $FH_ROBOT_CONFIG, else configs/real/robot.json)')
    a = ap.parse_args(argv)
    version, hashes = bundle_info()
    print(json.dumps(dict(bundle=version, sha256=hashes), indent=2), flush=True)
    if not (a.deploy or a.selftest or a.compile_check):
        return 0
    robot = load_robot_config(a.robot_config)
    if robot.is_example or robot.placeholders():
        ap.error('fill in configs/real/robot.json first (copy configs/real/robot.example.json): '
                 + ', '.join(robot.placeholders() or ['robot_pc.*']))
    if a.deploy:
        subprocess.run(ssh_argv(robot, deploy_command(robot, version)), input=archive(hashes), check=True, timeout=30)
    if a.selftest:
        urdf = robot.robot_pc.urdf
        subprocess.run(ssh_argv(robot, bundle_command(robot, 'selftest.py', ['--urdf', remote_path(urdf)] if urdf else [])),
                       check=True, timeout=60)
    if a.compile_check:
        subprocess.run(ssh_argv(robot, worker_command(BackendConfig(), robot, compile_only=True)), check=True, timeout=45)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
