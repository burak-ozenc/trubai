import json
from collections import Counter

with open("pitch_pairs_log.json") as f:
    log = json.load(f)

register_counts = Counter(
    entry["register"]          # adjust key name to match your actual log structure
    for entry in log
    if entry.get("shifts_written")
)
print(register_counts)s