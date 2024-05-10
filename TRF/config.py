from pathlib import Path


def get_config():
    return {
        "batch_size": 6,  # кол. параметров в итерации
        "num_epochs": 30,  # Кол. эпох
        "lr": 10**-4,
        "seq_len": 350,  # Размерность последовательности
        "d_model": 512,  # Размерность модели
        "datasource": "opus_books",  # Источник данных
        "lang_src": "en",  # Перевод с (с русского нельзя в этом словаре)
        "lang_tgt": "ru",  # Перевод на
        "model_folder": "weights",  # Куда сохранить
        "model_basename": "tmodel_",  # Префикс
        "preload": None,  # Версия "latest" | None
        "tokenizer_file": "tokenizer_{0}.json",  # Файл с токенами языка
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
