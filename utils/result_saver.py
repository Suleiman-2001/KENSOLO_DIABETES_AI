# utils/result_saver.py
import os, json

def save_results(output, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/predictions.json", "w") as f:
        json.dump(output.get("predictions", {}), f, indent=4)
    with open(f"{folder}/recommendations.json", "w") as f:
        json.dump(output.get("recommendations", {}), f, indent=4)
