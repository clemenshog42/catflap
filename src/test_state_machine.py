import unittest
from state_machine import StateMachine, State

class TestStateMachine(unittest.TestCase):
    def setUp(self):
        # Shorter history for easier testing
        self.sm = StateMachine(history_length=5, threshold=0.6, max_missed_frames=10)

    def test_initial_state(self):
        # A new track should start in CAT_NO_PREY or NO_CAT depending on the first confidence
        state = self.sm.update(track_id=1, prey_confidence=0.1, frame_idx=1)
        self.assertEqual(state, State.CAT_NO_PREY)
        
    def test_transition_to_prey(self):
        # Send 5 frames of high confidence
        for i in range(1, 6):
            state = self.sm.update(track_id=1, prey_confidence=0.8, frame_idx=i)
            
        # The moving average of five 0.8s is 0.8, which is >= 0.6
        self.assertEqual(state, State.CAT_WITH_PREY)
        
    def test_fluctuating_confidence(self):
        # History size is 5
        self.sm.update(1, 0.9, 1)
        self.sm.update(1, 0.9, 2)
        self.sm.update(1, 0.1, 3)
        self.sm.update(1, 0.1, 4)
        state = self.sm.update(1, 0.1, 5)
        # Average is (0.9+0.9+0.1+0.1+0.1)/5 = 0.42 < 0.6
        self.assertEqual(state, State.CAT_NO_PREY)
        
    def test_cleanup_stale_tracks(self):
        self.sm.update(track_id=1, prey_confidence=0.5, frame_idx=1)
        self.assertIn(1, self.sm.tracks)
        
        # Frame 15: 15 - 1 = 14 missed frames > max_missed_frames (10)
        self.sm.cleanup_stale_tracks(current_frame=15)
        self.assertNotIn(1, self.sm.tracks)

if __name__ == '__main__':
    unittest.main()
