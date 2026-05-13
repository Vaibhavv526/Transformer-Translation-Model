from pathlib import Path

def get_config():
    return{
    "batch_size": 8,
    "num_epochs": 65,
    "lr": 1e-4,
    "seq_len": 128,
    "d_model": 256,
    "lang_src": "en",
    "lang_tgt" : "hi",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": "54",
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"


    }

def get_weight_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder/ model_filename)

def latest_weights_file_path(config):

    model_folder = config['model_folder']
    model_basename = config['model_basename']

    model_files = list(
        Path(model_folder).glob(f"{model_basename}*.pt")
    )

    if len(model_files) == 0:
        return None

    model_files.sort()

    return str(model_files[-1])