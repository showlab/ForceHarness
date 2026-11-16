"""Custom contact-rich environments of HiddenPhys (robosuite 1.5 + MuJoCo).

PegInHoleRound: a block with a round hole is fixed on the table (a HollowCylinderObject whose inner radius is the half
diagonal of the peg plus the clearance), and a square peg starts in the gripper. The task tests insertion only, not
grasping.
ComSortBar: a bar whose centre of mass is either centred or offset along its long axis by a hidden ballast.
"""
from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, HollowCylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.placement_samplers import UniformRandomSampler


class PegInHoleRound(Lift):
    def __init__(self, *args, peg_radius: float = 0.0125, peg_half_length: float = 0.03, clearance: float = 0.0045,
                 hole_height: float = 0.04, hole_outer_radius: float = 0.09, hole_xy_range=((0.05, 0.20), (-0.15, 0.15)),
                 floor_depth=None, **kwargs):
        # floor_depth: optional hidden inner floor, floor_depth metres below the hole top (None = the hole goes down to the
        # table). Not used by the HiddenPhys tasks.
        self.floor_depth = floor_depth
        self.peg_radius, self.peg_half_length, self.clearance = peg_radius, peg_half_length, clearance
        self.hole_height, self.hole_outer_radius, self.hole_xy_range = hole_height, hole_outer_radius, hole_xy_range
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # Square peg: a cylinder or capsule held between two finger pads swings about the finger axis and its tip lands
        # 2-3 cm off, while a square peg rests on faces and does not swing. Hole radius = half diagonal of the peg + clearance.
        self.peg_half_width = self.peg_radius / 2 ** 0.5
        self.peg = BoxObject(name="peg", size=[self.peg_half_width, self.peg_half_width, self.peg_half_length], rgba=[0.15, 0.25, 0.85, 1],
                             density=1200, friction=[1.0, 0.005, 0.0001])
        self.hole = HollowCylinderObject(name="hole", outer_radius=self.hole_outer_radius, inner_radius=self.peg_radius + self.clearance,
                                         height=self.hole_height, ngeoms=24, rgba=[0.55, 0.55, 0.5, 1], density=3000)
        body = self.hole.get_obj()
        for j in list(body.findall("joint")):      # the hole block is welded to the table: remove the free joint from the XML
            body.remove(j)
        for attr in ("_joints", "joint_specs"):   # robosuite 1.5's joints is a read-only property; change the private fields behind it
            if hasattr(self.hole, attr):
                try:
                    setattr(self.hole, attr, [])
                except Exception:
                    pass
        body.set("pos", array_to_string([0.12, 0.0, self.table_offset[2] + self.hole_height / 2]))
        if self.floor_depth is not None:           # inner floor: a solid cylinder from the bottom of the block up to floor_depth below the true opening (a hard stop)
            # Note: HollowCylinderObject's height is a half-height, so the block spans body centre +/- hole_height and the
            # true opening is centre + hole_height. hole_top() uses centre + hole_height / 2, i.e. hole_height / 2 too low;
            # the peg tasks keep hole_top() (see hole_top_true).
            import xml.etree.ElementTree as ET
            fd = float(self.floor_depth)
            ET.SubElement(body, "geom", attrib=dict(name="hole_floor", type="cylinder",
                                                    size=f"{self.peg_radius + self.clearance - 0.0005} {(2 * self.hole_height - fd) / 2}",
                                                    pos=f"0 0 {-fd / 2}", rgba="0.25 0.25 0.25 1",
                                                    group="0", contype="1", conaffinity="1", friction="1 0.005 0.0001",
                                                    # hard contact: the peg weighs only 22 g and the default solref scales contact
                                                    # stiffness with mass, so 10 N would push it 4.5 mm into the floor
                                                    solref="-20000 -150"))
        self.cube = self.peg                       # reuse Lift's references (cube_body_id etc.)
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.peg, x_range=[-0.25, -0.2], y_range=[0.2, 0.25], rotation=0.0,
            rotation_axis="z", ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.02)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[self.peg, self.hole])
        # "The peg is already in the hand" is a weld constraint (activated at reset, removed by the release primitive).
        # robosuite's position-controlled fingers (kp = 1000) squeeze the 17.7 mm square peg with only 8 N each, so in
        # simulation the peg slowly slips between the pads and the wrist cannot feel the support force during a search;
        # a real Franka Hand holds 70 N continuously.
        import xml.etree.ElementTree as ET
        try:
            grip_body = self.robots[0].gripper["right"].root_body
        except Exception:
            grip_body = "gripper0_right_right_gripper"
        eq = self.model.root.find("equality")
        if eq is None:
            eq = ET.SubElement(self.model.root, "equality")
        ET.SubElement(eq, "weld", attrib=dict(name="held_weld", body1=grip_body, body2=self.peg.root_body, active="false"))
        self.held_weld_name = "held_weld"

    def _setup_references(self):
        super()._setup_references()
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def hole_top(self) -> np.ndarray:
        p = np.array(self.sim.data.body_xpos[self.hole_body_id])
        return p + np.array([0, 0, self.hole_height / 2])

    def hole_top_true(self) -> np.ndarray:
        """The true hole opening. hole_top() is hole_height / 2 too low because the height of a HollowCylinderObject is a
        half-height; the peg tasks keep hole_top() for their scene and success check, this is for variants that need the
        true opening."""
        p = np.array(self.sim.data.body_xpos[self.hole_body_id])
        return p + np.array([0, 0, self.hole_height])

    def _check_success(self):
        peg = np.array(self.sim.data.body_xpos[self.peg_body_id])
        top = self.hole_top()
        xy_ok = np.linalg.norm(peg[:2] - top[:2]) < self.clearance + 0.003
        depth = top[2] - (peg[2] - self.peg_half_length)          # how far the peg bottom is below the hole top
        return bool(xy_ok and depth > 0.02)

    def reward(self, action=None):
        return float(self._check_success())


class ComSortBar(Lift):
    """A bar (12 x 4 x 4 cm, 0.8 kg by default) whose centre of mass is either centred or offset along its long axis
    (``com_offset``, fixed when the model is built). Lifted at its geometric centre, an offset bar produces a moment of
    about m g d at the wrist (0.31 N*m for 0.8 kg and 4 cm): a camera cannot see it, only the moment tells."""

    def __init__(self, *args, bar_half=(0.06, 0.02, 0.02), bar_mass=0.8, com_offset=0.0, **kwargs):
        self.bar_half, self.bar_mass, self.com_offset = bar_half, bar_mass, float(com_offset)
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # Centre-of-mass offset: two geoms in one body, a light shell and a dense hidden ballast at the offset, so the centre
        # of mass is offset at compile time (changing body_ipos at run time or appending a geom to the XML has no effect in
        # robosuite).
        from robosuite.models.objects import CompositeObject
        shell_mass, ballast_mass = self.bar_mass * 0.25, self.bar_mass * 0.75
        vol = 8 * self.bar_half[0] * self.bar_half[1] * self.bar_half[2]
        w_half = 0.008
        w_pos = self.com_offset * self.bar_mass / ballast_mass        # total centre of mass = com_offset
        self.bar = CompositeObject(
            name="bar", total_size=list(self.bar_half), geom_types=["box", "box"],
            geom_sizes=[list(self.bar_half), [w_half, w_half, w_half]],
            geom_locations=[[0.0, 0.0, 0.0], [w_pos, 0.0, 0.0]], locations_relative_to_center=True,
            geom_names=["shell", "ballast"], geom_rgbas=[[0.8, 0.6, 0.2, 1.0], [0.8, 0.6, 0.2, 0.0]],
            geom_frictions=[[1.0, 0.005, 0.0001], [1.0, 0.005, 0.0001]],
            density=[shell_mass / vol, ballast_mass / (2 * w_half) ** 3], joints="default")
        self.cube = self.bar
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.bar, x_range=[-0.03, 0.03], y_range=[-0.03, 0.03], rotation=(-0.6, 0.6),
            rotation_axis="z", ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[self.bar])

    def _setup_references(self):
        super()._setup_references()
        self.bar_body_id = self.sim.model.body_name2id(self.bar.root_body)
