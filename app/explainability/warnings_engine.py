import yaml

def load_warnings(path):
    with open(path) as f:
        return yaml.safe_load(f)["warnings"]

def build_warnings(biomech, warnings):
    triggered = []

    for w in warnings:
        ok = True
        for k, v in w["when"].items():
            if biomech.get(k) != v:
                ok = False
                break
        if ok:
            triggered.append({
                "area": w["area"],
                "severity": w["severity"],
                "explanation": w["explanation"]
            })

    return triggered
