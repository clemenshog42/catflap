import io

file_path = 'src/state_machine.py'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = """        for track in self.tracks.values():
            if len(track.confidence_history) < self.history_length / 2:
                # We need at least half of the history to confirm a cat is actually clean
                all_cats_clean = False
            elif track.current_state == State.CAT_WITH_PREY:
                has_prey = True
                all_cats_clean = False"""

replacement = """        for track in self.tracks.values():
            if track.current_state == State.CAT_WITH_PREY:
                has_prey = True
                all_cats_clean = False
            if len(track.confidence_history) < self.history_length / 2:
                # We need at least half of the history to confirm a cat is actually clean
                all_cats_clean = False"""

new_text = text.replace(target, replacement)
with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Updated state_machine.py logic')
