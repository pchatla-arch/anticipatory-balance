import unittest
from foresight.adaptive.language_nav import Landmark
from foresight.adaptive.task_stack import LanguageGuidedNavigator

class FakeNavigator:
    def __init__(self, goal):
        self.goal=goal
        self.samples=[]
    def ingest(self, detection, observed_step):
        self.samples.append((detection, observed_step))
    def command(self, x,y,yaw,step):
        return .2,.1,"fake_detour"

class TaskStackTests(unittest.TestCase):
    def setUp(self):
        self.stack=LanguageGuidedNavigator(navigator_factory=FakeNavigator)
        self.landmarks=[Landmark('station',5.,2.,.97,10)]

    def test_grounded_task_drives_navigation(self):
        status=self.stack.set_language_goal('Walk to the station',self.landmarks,step=12)
        self.assertEqual(status,'grounded')
        self.stack.update_depth(None,observed_step=12)
        self.assertEqual(self.stack.navigator.samples,[(None,12)])
        self.assertEqual(self.stack.command(0.,0.,0.,13),(.2,.1,'fake_detour'))

    def test_invalid_instruction_stops(self):
        self.stack.set_language_goal('move to station',self.landmarks,step=12)
        self.stack.set_language_goal('jump over a box',self.landmarks,step=13)
        self.assertEqual(self.stack.command(0.,0.,0.,13),(0.,0.,'stop_no_grounded_goal'))

    def test_stale_goal_stops(self):
        self.assertEqual(self.stack.set_language_goal('walk to station',self.landmarks,step=30),'stale_or_future_landmark')
        self.assertEqual(self.stack.command(0.,0.,0.,31)[0],0.)

    def test_model_proposal_accepted_only_when_grounded(self):
        r=self.stack.set_model_goal({'operation':'navigate','target_id':'station','confidence':.94},self.landmarks,12)
        self.assertEqual(r,'grounded')
        r=self.stack.set_model_goal({'operation':'navigate','target_id':'station','confidence':.94,'actuator':1.4},self.landmarks,13)
        self.assertNotEqual(r,'grounded')
        self.assertIsNone(self.stack.navigator)

    def test_no_model_direct_motor_controls(self):
        r=self.stack.set_model_goal({'operation':'actuator_gain','target_id':'station','confidence':1.0},self.landmarks,12)
        self.assertEqual(r,'unsupported_model_action')
        self.assertEqual(self.stack.command(0,0,0,13)[2],'stop_no_grounded_goal')

if __name__=='__main__': unittest.main()
