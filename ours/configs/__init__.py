import platform

if "Windows" in platform.platform():
    from local_config import DataConfig, ModelConfig, TrainConfig
else:
    from server_config import DataConfig, ModelConfig, TrainConfig
