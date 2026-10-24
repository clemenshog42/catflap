import unittest
from state_machine import StateMachine, State, AccessState

class TestStateMachine(unittest.TestCase):
    def setUp(self):
        # Shorter history for easier testing
        self.sm = StateMachine(history_length=5, threshold=0.6, max_missed_frames=10)

    def test_initial_state(self):
        # A new clean track has to collect its full history before it counts as clean
        state = self.sm.update(track_id=1, prey_confidence=0.1, frame_idx=1)
        self.assertEqual(state, State.GATHERING_HISTORY)
        
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
        self.sm.process_global_state(current_frame=15, current_time=15.0)
        self.assertNotIn(1, self.sm.tracks)

    def test_invalid_frames_are_not_added(self):
        # Frames whose segmentation crop leaves the image do not change the history or the state
        for i in range(1, 6):
            self.sm.update(1, 0.9, i)
        self.assertEqual(self.sm.get_state(1), State.CAT_WITH_PREY)
        for i in range(6, 20):
            state = self.sm.update(1, 0.0, i, valid=False)
        self.assertEqual(state, State.CAT_WITH_PREY)
        self.assertEqual(list(self.sm.tracks[1].confidence_history), [0.9] * 5)
        self.assertEqual(self.sm.tracks[1].last_seen, 19)

    def _decisions(self):
        decisions = []
        self.sm.subscribe(decisions.append)
        return decisions

    def test_ghost_track_does_not_block_release(self):
        # A short-lived track (e.g. after an ID switch) that is no longer visible must not block the release
        decisions = self._decisions()
        self.sm.update(track_id=9, prey_confidence=0.0, frame_idx=1)
        self.sm.process_global_state(1, current_time=1.0)
        for i in range(2, 7):
            self.sm.update(track_id=1, prey_confidence=0.0, frame_idx=i)
            self.sm.process_global_state(i, current_time=float(i))
        self.assertIn(9, self.sm.tracks)
        self.assertEqual(decisions, [AccessState.GRANTED])

    def test_hidden_prey_track_keeps_flap_locked(self):
        # A cat with prey keeps the flap locked until its track is removed, even if it is not visible
        decisions = self._decisions()
        for i in range(1, 6):
            self.sm.update(track_id=9, prey_confidence=0.9, frame_idx=i)
            self.sm.process_global_state(i, current_time=float(i))
        for i in range(6, 12):
            self.sm.update(track_id=1, prey_confidence=0.0, frame_idx=i)
            self.sm.process_global_state(i, current_time=float(i))
        self.assertEqual(decisions, [AccessState.DENIED])

    def test_no_release_without_clean_cat(self):
        # An incomplete track that is not visible in this frame must not lead to a release on its own
        decisions = self._decisions()
        self.sm.update(track_id=1, prey_confidence=0.0, frame_idx=1)
        self.sm.process_global_state(1, current_time=1.0)
        self.sm.process_global_state(2, current_time=2.0)
        self.assertEqual(decisions, [])

if __name__ == '__main__':
    unittest.main()
