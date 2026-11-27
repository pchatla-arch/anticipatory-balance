"""Pure-stdlib physical stationary-cylinder scene builder for G1.

This modifies the *scene XML*, not simulator state after rollout. The static
obstacle is actually collision-enabled via explicit MuJoCo contact pairs with
the known torso/pelvis/thigh geoms, matching the project's feet-only G1
contact convention (robot geoms normally have contype=0).
Ground-truth obstacle geometry must remain evaluator-only, never fed into
the depth detector or online planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class StaticCylinder:
    x: float = 2.8
    y: float = 0.0
    radius_m: float = 0.30
    height_m: float = 1.05

    def validate(self):
        if not all(isfinite(v) for v in (self.x, self.y, self.radius_m, self.height_m)):
            raise ValueError("nonfinite cylinder geometry")
        if not (0.10 <= self.radius_m <= 0.5 and 0.4 <= self.height_m <= 1.8):
            raise ValueError("unsupported stationary cylinder dimensions")
        if hypot(self.x, self.y) < self.radius_m + 1.2 or hypot(self.x, self.y) > 8.0:
            raise ValueError("cylinder must be away from initial G1 pose and within test course")
        return self


COLLISION_TARGETS = (
    ("torso_impact_geom", "torso"),
    ("pelvis_impact_geom", "pelvis"),
    ("left_thigh", "left_thigh"),
    ("right_thigh", "right_thigh"),
)
OBSTACLE_GEOM = "g1_static_obstacle_geom"


def add_static_cylinder(scene_xml: str, spec: StaticCylinder) -> str:
    """Inject static physical cylinder with explicit torso/pelvis/thigh pairs.

    No floating joint or actuator is added: all previous action/observation
    dimensions and checkpoint inference remain unchanged. Preserves the
    existing projectile scene and static calibration camera.
    """
    spec.validate()
    root = ET.fromstring(scene_xml)
    world = root.find("worldbody")
    contact = root.find("contact")
    sensor = root.find("sensor")
    if world is None or contact is None or sensor is None:
        raise ValueError("G1 scene missing required world/contact/sensor section")
    if root.find(".//geom[@name='" + OBSTACLE_GEOM + "']") is not None:
        raise ValueError("stationary cylinder already injected")

    ET.SubElement(world, "geom", {
        "name": OBSTACLE_GEOM,
        "type": "cylinder",
        "pos": f"{spec.x:.6f} {spec.y:.6f} {spec.height_m/2:.6f}",
        "size": f"{spec.radius_m:.6f} {spec.height_m/2:.6f}",
        "rgba": "0.12 0.26 0.88 1",
        "contype": "0",
        "conaffinity": "0",
        "solref": "0.005 1",
        "solimp": "0.95 0.99 0.001",
        "group": "0",
    })
    for geom, tag in COLLISION_TARGETS:
        pair_name = "g1_obstacle_" + tag
        ET.SubElement(contact, "pair", {
            "name": pair_name,
            "geom1": OBSTACLE_GEOM,
            "geom2": geom,
            "condim": "3",
            "friction": "0.6 0.6",
        })
        ET.SubElement(sensor, "contact", {
            "name": pair_name + "_found",
            "geom1": OBSTACLE_GEOM,
            "geom2": geom,
            "reduce": "mindist",
            "num": "1",
            "data": "found",
        })
    return ET.tostring(root, encoding="unicode") + "\n"
