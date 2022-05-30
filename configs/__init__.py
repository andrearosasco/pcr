import platform

print(platform.platform())
if not "Windows" in platform.platform():
    from .local_config import DataConfig, ModelConfig, TrainConfig
else:
    from .server_config import DataConfig, ModelConfig, TrainConfig
