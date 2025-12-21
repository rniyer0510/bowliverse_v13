import yaml
from pathlib import Path

CUES_PATH = Path(__file__).parent / "cues.yaml"


def load_cues():
    with open(CUES_PATH, "r") as f:
        return yaml.safe_load(f)


def classify_posture(biomech):
    sh = biomech.get("shoulder_hip", {}).get("zone")
    bf = biomech.get("backfoot", {}).get("zone")
    if sh == "LOW" or bf == "VERY_OPEN":
        return "LOW"
    if sh == "MIXED":
        return "MEDIUM"
    return "GOOD"


def classify_power(risks):
    for r in risks:
        if r.get("id") == "FRONT_FOOT_BRAKING" and r.get("level") in ("MEDIUM", "HIGH"):
            return "STRONG"
    return "BALANCED"


def classify_protection(risks):
    for r in risks:
        if r.get("level") in ("MEDIUM", "HIGH"):
            return "LOW"
    return "GOOD"


def build_cues(ctx):
    biomech = ctx.biomech
    risks = biomech.get("risk", {}).get("breakdown", [])

    posture = classify_posture(biomech)
    power = classify_power(risks)
    protection = classify_protection(risks)

    key = f"{posture}_{power}_{protection}"
    cues_map = load_cues()

    selected = cues_map.get(key, [])

    return {
        "state": key,
        "list": selected,
    }
