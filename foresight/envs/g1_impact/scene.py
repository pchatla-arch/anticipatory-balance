# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Builds the G1 impact scene on top of the installed mujoco_playground (rss2025) G1 model.

The Playground feet-only G1 has no torso/pelvis collision geometry and every geom has
contype/conaffinity 0, so contacts exist only through explicit <pair> entries. We therefore
(1) inject two collision capsules into the robot xml string, and (2) add pairs between the
projectile and those capsules in the scene. Nothing in the Playground checkout is modified.
"""
from __future__ import annotations

import re
from pathlib import Path

from etils import epath

try:
  from mujoco_playground._src.locomotion.g1 import g1_constants as consts
except ImportError as e:  # pragma: no cover
  raise ImportError(
      "mujoco_playground (rss2025) must be installed; see docs/setup.md"
  ) from e

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "xmls" / "scene_impact_flat.xml.tmpl"

# Capsules approximating the torso and pelvis; sizes from the G1 torso_link mesh extents.
TORSO_CAPSULE = (
    '<geom name="torso_impact_geom" class="collision" type="capsule" '
    'size="0.09" fromto="0 0 0.02 0 0 0.30"/>\n'
    '<site name="torso_impact_site" pos="0 0 0.16" size="0.1" group="5"/>'
)
PELVIS_CAPSULE = (
    '<geom name="pelvis_impact_geom" class="collision" type="capsule" '
    'size="0.08" fromto="0 -0.06 0 0 0.06 0"/>'
)


def _inject_capsules(robot_xml: str) -> str:
  """Insert collision capsules right after the torso_link / pelvis inertial tags."""
  robot_xml, n1 = re.subn(
      r'(<body name="torso_link">\s*<inertial[^>]*/>)',
      r"\1\n            " + TORSO_CAPSULE,
      robot_xml,
      count=1,
  )
  robot_xml, n2 = re.subn(
      r'(<body name="pelvis"[^>]*>\s*<inertial[^>]*/>)',
      r"\1\n      " + PELVIS_CAPSULE,
      robot_xml,
      count=1,
  )
  if n1 != 1 or n2 != 1:
    raise RuntimeError(
        f"capsule injection failed (torso={n1}, pelvis={n2}); "
        "the Playground G1 xml layout may have changed"
    )
  return robot_xml


def build_scene(
    out_dir: Path,
    proj_radius: float = 0.15,
    proj_mass: float = 4.0,
    park_x: float = 6.0,
    park_z: float = 0.9,
) -> tuple[Path, dict[str, str]]:
  """Write a patched robot xml and the impact scene xml into out_dir.

  Returns the scene path and the asset dict expected by G1Env (mesh files resolved
  relative to Playground's xml dir, which is what the included robot xml assumes).
  """
  g1_dir = epath.Path(consts.ROOT_PATH) / "xmls"
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  robot_src = (g1_dir / "g1_mjx_feetonly.xml").read_text()
  robot_patched = _inject_capsules(robot_src)
  # Mesh paths in the robot xml are relative to Playground's xml dir; make them absolute.
  menagerie_rel = "../../../../../mujoco_menagerie/unitree_g1/assets/"
  menagerie_abs = str((g1_dir / menagerie_rel).resolve()) + "/"
  robot_patched = robot_patched.replace(menagerie_rel, menagerie_abs)
  robot_path = out_dir / "g1_impact_robot.xml"
  robot_path.write_text(robot_patched)

  scene = TEMPLATE.read_text().format(
      ROBOT_XML=str(robot_path),
      G1_XML_DIR=str(g1_dir),
      PROJ_RADIUS=proj_radius,
      PROJ_MASS=proj_mass,
      PARK_X=park_x,
      PARK_Z=park_z,
  )
  scene_path = out_dir / "scene_impact_flat.xml"
  scene_path.write_text(scene)
  return scene_path, {}
