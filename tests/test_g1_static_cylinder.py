"""CPU-only contract for physical static obstacle MJCF scene injection."""
import unittest
from xml.etree import ElementTree as ET

from foresight.adaptive.obstacle_scene_xml import (
    StaticCylinder, add_static_cylinder, OBSTACLE_GEOM, COLLISION_TARGETS,
)

BASE = """<mujoco model="g1">
  <include file="/tmp/g1_robot.xml"/>
  <worldbody><geom name="floor" type="plane" size="0 0 0.01"/></worldbody>
  <contact><pair name="old_projectile" geom1="projectile_geom" geom2="torso_impact_geom"/></contact>
  <sensor><contact name="old_projectile_found" geom1="projectile_geom"
                   geom2="torso_impact_geom" reduce="mindist" num="1" data="found"/></sensor>
</mujoco>"""


class StaticObstacleSceneTests(unittest.TestCase):
    def test_enabled_geometry_stays_static_and_collidable(self):
        xml = add_static_cylinder(BASE, StaticCylinder(x=2.8, y=0.0))
        root = ET.fromstring(xml)
        geom = root.find(".//geom[@name='" + OBSTACLE_GEOM + "']")
        self.assertIsNotNone(geom)
        self.assertEqual(geom.attrib["type"], "cylinder")
        self.assertEqual(geom.attrib["size"], "0.300000 0.525000")
        self.assertEqual(geom.attrib["pos"], "2.800000 0.000000 0.525000")
        self.assertIsNone(root.find(".//body[@name='g1_static_obstacle']"))
        # Existing projectile contacts remain and physical G1 contact pairs are explicit.
        pairs = root.findall("./contact/pair")
        self.assertEqual(len(pairs), len(COLLISION_TARGETS) + 1)
        for robot_geom, tag in COLLISION_TARGETS:
            pair = root.find("./contact/pair[@name='g1_obstacle_" + tag + "']")
            self.assertEqual(pair.attrib["geom1"], OBSTACLE_GEOM)
            self.assertEqual(pair.attrib["geom2"], robot_geom)
            sensor = root.find("./sensor/contact[@name='g1_obstacle_" + tag + "_found']")
            self.assertEqual(sensor.attrib["data"], "found")
            self.assertEqual(sensor.attrib["reduce"], "mindist")

    def test_no_privileged_obstacle_coordinate_escapes_into_navigation(self):
        from foresight.adaptive.obstacle_depth import detect_obstacle_points
        import inspect
        self.assertNotIn("g1_static_obstacle_geom", inspect.getsource(detect_obstacle_points))

    def test_preserve_include_and_unrelated_contacts(self):
        root = ET.fromstring(add_static_cylinder(BASE, StaticCylinder()))
        self.assertEqual(root.find("include").attrib["file"], "/tmp/g1_robot.xml")
        self.assertIsNotNone(root.find("./contact/pair[@name='old_projectile']"))

    def test_reinjection_rejected(self):
        xml = add_static_cylinder(BASE, StaticCylinder())
        with self.assertRaisesRegex(ValueError, "already injected"):
            add_static_cylinder(xml, StaticCylinder())

    def test_reject_geometry_near_robot_or_nonfinite(self):
        for obj in (
            StaticCylinder(x=0.0), StaticCylinder(x=0.9),
            StaticCylinder(x=float("nan")), StaticCylinder(radius_m=-0.4),
            StaticCylinder(height_m=0.01), StaticCylinder(x=20),
        ):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                add_static_cylinder(BASE, obj)

    def test_require_contact_and_sensor_sections(self):
        for missing in ("<contact>", "<sensor>", "<worldbody>"):
            s = BASE
            if missing == "<contact>":
                s = s.replace('<contact><pair name="old_projectile" geom1="projectile_geom" geom2="torso_impact_geom"/></contact>', "")
            elif missing == "<sensor>":
                s = s.replace('<sensor><contact name="old_projectile_found" geom1="projectile_geom"\n                   geom2="torso_impact_geom" reduce="mindist" num="1" data="found"/></sensor>', "")
            else:
                s = s.replace('<worldbody><geom name="floor" type="plane" size="0 0 0.01"/></worldbody>', "")
            with self.subTest(section=missing), self.assertRaisesRegex(ValueError, "missing required"):
                add_static_cylinder(s, StaticCylinder())


if __name__ == "__main__":
    unittest.main()
