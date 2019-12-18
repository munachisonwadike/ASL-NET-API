import numpy as np

# Parses yolo configuration file, determining network architecture 
def model_cfg_parser(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [line for line in lines if line and not line.startswith('#')]
    lines = [line.rstrip().lstrip() for line in lines]  
    modules = []   
    for ln in lines:
        # Each new block is identified with brackets []
        if ln.startswith('['):   
            modules.append({})
            modules[-1]['type'] = ln[1:-1].rstrip()
            if modules[-1]['type'] == 'convolutional':
                modules[-1]['batch_normalize'] = 0  
        # If we are not at a new block, we are still in the old one
        else:
            k, value = ln.split("=")
            k = k.rstrip()

            if 'anchors' in k:
                modules[-1][k] = np.array([float(x) for x in value.split(',')]).reshape((-1, 2))  
            else:
                modules[-1][k] = value.strip()

    return modules

# Parses the .data configuration file
def data_cfg_parser(path):
    conf = {}
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for ln in lines:
        ln = ln.strip()
        if ln == '' or ln.startswith('#'): #ignore comments
            continue
        k, value = ln.split('=')
        conf[k.strip()] = value.strip()

    return conf
