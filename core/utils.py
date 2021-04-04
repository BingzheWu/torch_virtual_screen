import yaml

def load_yaml_cfg(args_file):
    with open(args_file, 'r') as f:
        args = yaml.load(f)
    return args