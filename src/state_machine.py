from enum import Enum
from collections import deque
import numpy as np

class State(Enum):
    NO_CAT = "NO_CAT"
    CAT_NO_PREY = "CAT_NO_PREY"
    CAT_WITH_PREY = "CAT_WITH_PREY"

class TrackState:
    """Manages the state and confidence history for a single tracked object (cat)."""
    
    def __init__(self, track_id, history_length=15, threshold=0.2):
        self.track_id = track_id
        # Queue to hold the last 'history_length' confidence scores for prey
        self.confidence_history = deque(maxlen=history_length)
        self.threshold = threshold
        self.current_state = State.CAT_NO_PREY
        self.last_seen = 0 # Frame counter or timestamp
        
    def update(self, prey_confidence, frame_idx):
        """Update track history with a new confidence score."""
        self.confidence_history.append(prey_confidence)
        self.last_seen = frame_idx
        
        # Calculate moving average
        avg_confidence = np.mean(self.confidence_history)
        
        # Determine new state based on aggregated confidence
        if avg_confidence >= self.threshold:
            self.current_state = State.CAT_WITH_PREY
        else:
            self.current_state = State.CAT_NO_PREY
            
        return self.current_state

class StateMachine:
    """Manages multiple TrackStates and cleans up stale tracks."""
    
    def __init__(self, history_length=15, threshold=0.8, max_missed_frames=30):
        self.tracks = {} # track_id -> TrackState
        self.history_length = history_length
        self.threshold = threshold
        self.max_missed_frames = max_missed_frames
        
    def update(self, track_id, prey_confidence, frame_idx):
        """Update a specific track and return its state."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                track_id, 
                history_length=self.history_length, 
                threshold=self.threshold
            )
            
        return self.tracks[track_id].update(prey_confidence, frame_idx)
        
    def cleanup_stale_tracks(self, current_frame):
        """Remove tracks that haven't been seen for a while."""
        stale_ids = []
        for track_id, track in self.tracks.items():
            if current_frame - track.last_seen > self.max_missed_frames:
                stale_ids.append(track_id)
                
        for tid in stale_ids:
            del self.tracks[tid]
            
    def get_state(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id].current_state
        return State.NO_CAT
