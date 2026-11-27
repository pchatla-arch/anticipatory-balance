#!/usr/bin/env python3
"""D1/D3 MuJoCo contact smoke for the physically added static cylinder.

Two modes:
  fixture -- an isolated tiny articulated-body *physics fixture*, not G1
  g1      -- compile the real included G1 scene from pinned Playground assets

The close-contact test deliberately moves the stationary obstacle into the
robot in an evaluation-only throwaway MjModel. Those model coordinates must
NEVER be provided to online perception/navigation or adaptation methods.
"""
import argparse
import json
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET

from foresight.adaptive.obstacle_scene_xml import (
    StaticCylinder, add_static_cylinder, OBSTACLE_GEOM, COLLISION_TARGETS,
)

# The fixture has real masses, gravity and a free base. It is a unit physics
# test for collision pairs, and must not be presented as actual G1 locomotion.
TINY_SCENE = """<mujoco model="stationary obstacle contact fixture">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
    <body name="fixture_body" pos="2.8 0 0.65">
      <freejoint name="fixture_free"/>
      <geom name="torso_impact_geom" type="sphere" size="0.12" pos="0 0 0.32"
            mass="4" contype="0" conaffinity="0"/>
      <geom name="pelvis_impact_geom" type="sphere" size="0.13" pos="0 0 0.01"
            mass="4" contype="0" conaffinity="0"/>
      <geom name="left_thigh" type="sphere" size="0.10" pos="0 0.16 -0.15"
            mass="2" contype="0" conaffinity="0"/>
      <geom name="right_thigh" type="sphere" size="0.10" pos="0 -0.16 -0.15"
            mass="2" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact/>
  <sensor/>
</mujoco>"""


def collision_stats(model, data):
    import mujoco
    obstacle_id = model.geom(OBSTACLE_GEOM).id
    matches = []
    for k in range(data.ncon):
        contact = data.contact[k]
        if obstacle_id in (contact.geom1, contact.geom2):
            matches.append({"geom1": model.geom(contact.geom1).name,
                            "geom2": model.geom(contact.geom2).name,
                            "distance": float(contact.dist)})
    return matches


def run_fixture():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        add_static_cylinder(TINY_SCENE, StaticCylinder()))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    collisions = collision_stats(model, state)
    if not collisions:
        raise RuntimeError("Fixture cylinder did not make physical contact")
    return {"mode": "fixture", "contact_pairs": len(COLLISION_TARGETS),
            "contacts_observed": len(collisions), "contacts": collisions,
            "warning": "Fixture-only physical MuJoCo test; not G1 locomotion"}


def run_g1():
    import mujoco
    from foresight.envs.g1_impact import scene
    with tempfile.TemporaryDirectory(prefix="g1_static_physics_") as tmp:
        scene_path, _ = scene.build_scene(
            Path(tmp), static_obstacle={
                "x": 2.8, "y": 0.0, "radius_m": 0.30, "height_m": 1.05})
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        # A compile check verifies the actual pinned G1 world asset, camera,
        # obstacle contact pairs and sensor names, not just the fixture XML.
        if model.camera("head_cam").id < 0:
            raise RuntimeError("G1 head camera missing")
        obstacle = model.geom(OBSTACLE_GEOM).id
        pairs = [i for i in range(model.npair)
                 if obstacle in (int(model.pair_geom1[i]),
                                 int(model.pair_geom2[i]))]
        if len(pairs) < len(COLLISION_TARGETS):
            raise RuntimeError("G1 static obstacle lacks contact-pair coverage")
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("knees_bent").id)
        mujoco.mj_forward(model, data)
        initial = collision_stats(model, data)
        # Proof that the G1 capsule and static cylinder physically collide:
        # force overlap in a separate throwaway evaluation state, NOT online.
        torso = model.geom("torso_impact_geom").id
        torso_xy = data.geom_xpos[torso, :2].copy()
        model.geom_pos[obstacle, :2] = torso_xy
        mujoco.mj_forward(model, data)
        forced = collision_stats(model, data)
        if not forced:
            raise RuntimeError("G1 static cylinder does not physically contact torso")
        return {"mode": "g1", "nq": int(model.nq), "nv": int(model.nv),
                "nu": int(model.nu), "cameras": int(model.ncam),
                "collision_pairs": len(pairs),
                "initial_obstacle_contacts": len(initial),
                "forced_overlap_contacts": len(forced),
                "warning": "Real G1 model collision smoke, NOT closed-loop walking"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "g1"), default="fixture")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_g1() if args.mode == "g1" else run_fixture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
