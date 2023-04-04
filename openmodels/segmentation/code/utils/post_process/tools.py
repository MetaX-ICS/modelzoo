import json


Jsonfile = "./%s/config.json"

def get_params(modelname):
    
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
        params = param_all["model_config"]
    return params#,commons_file

def preprocess_test(path):#,commons_file
    image_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip('\n').split())
    return image_list