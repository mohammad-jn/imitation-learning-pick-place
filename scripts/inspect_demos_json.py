import pickle
import json

with open("data/demos.pkl", "rb") as f:
    data = pickle.load(f)

# convert tuples → lists (JSON doesn't support tuples)
def convert(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(x) for x in obj]
    return obj

data = convert(data)

with open("data/demos.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved to data/demos.json")