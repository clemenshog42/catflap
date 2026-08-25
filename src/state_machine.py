from enum import Enum
from collections import deque
import numpy as np

class State(Enum):
    NO_CAT = "NO_CAT"
    CAT_NO_PREY = "CAT_NO_PREY"
    CAT_WITH_PREY = "CAT_WITH_PREY"

class AccessState(Enum):
    DENIED = "DENIED"
    GRANTED = "GRANTED"

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
        self.current_state = State.CAT_WITH_PREY if avg_confidence >= self.threshold else State.CAT_NO_PREY
            
        return self.current_state

class StateMachine:
    """Manages multiple TrackStates and evaluates the global access decision."""
    
    def __init__(self, history_length=15, threshold=0.8, max_missed_frames=30):
        self.tracks = {} # track_id -> TrackState
        self.history_length = history_length
        self.threshold = threshold
        self.max_missed_frames = max_missed_frames
        
        self.global_callbacks = []
        self.last_state = None
        
    def subscribe(self, callback):
        """Subscribe to global access state changes (AccessState.GRANTED / DENIED)."""
        self.global_callbacks.append(callback)
        
    def update(self, track_id, prey_confidence, frame_idx):
        """Update a specific track and return its state."""
        if track_id not in self.tracks:
            track = TrackState(
                track_id, 
                history_length=self.history_length, 
                threshold=self.threshold
            )
            self.tracks[track_id] = track
            
        return self.tracks[track_id].update(prey_confidence, frame_idx)
        
    def process_global_state(self, current_frame):
        """
        Evaluates all active tracks to make a global decision on access.
        Rule 1: If ANY cat has prey, DENY access.
        Rule 2: If NO cat has prey, and AT LEAST ONE cat is clean, GRANT access.
        Triggers callbacks ONLY when the global access state changes.
        """
        # First, remove stale tracks that haven't been seen recently
        stale_ids = []
        for track_id, track in self.tracks.items():
            if current_frame - track.last_seen > self.max_missed_frames:
                stale_ids.append(track_id)
                
        for tid in stale_ids:
            del self.tracks[tid]
            
        # Determine global state based on remaining active tracks
        has_prey = False
        has_clean_cat = False
        
        for track in self.tracks.values():
            if track.current_state == State.CAT_WITH_PREY:
                has_prey = True
            elif track.current_state == State.CAT_NO_PREY:
                has_clean_cat = True
                
        access_state = None
        if has_prey:
            # Immediate priority: deny access if any prey is present
            access_state = AccessState.DENIED
        elif has_clean_cat:
            # Safe to grant access if at least one clean cat is present and NO prey is present
            access_state = AccessState.GRANTED
            
        # Only notify if we reached a conclusive state AND it's different from the last emitted state
        if access_state and access_state != self.last_state:
            self.last_state = access_state
            for cb in self.global_callbacks:
                cb(access_state)
                
    def get_state(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id].current_state
        return State.NO_CAT
