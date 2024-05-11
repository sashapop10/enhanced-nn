from pathlib import Path


def get_config():
    return {
        "batch_size": 6,  # Parameters
        "num_epochs": 30,  
        "lr": 10**-4,
        "seq_len": 350,  # Max Tokens
        "d_model": 512,  # Embedding size / Matrix size
        "datasource": "opus_books",  # Dataset source
        "lang_src": "en",  # (Missing RU)
        "lang_tgt": "ru",  
        "model_folder": "weights",  # Save point
        "model_basename": "tmodel_", 
        "preload": "latest",  # "latest" | None
        "tokenizer_file": "tokenizer_{0}.json",  # Token file
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
