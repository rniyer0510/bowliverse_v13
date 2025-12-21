import yaml

def load_cues(path):
    with open(path) as f:
        return yaml.safe_load(f)["cues"]

def build_cues(state, cues_config):
    return cues_config.get(state, [])
