import platform

print(platform.platform())
# if "Windows" in platform.platform():
#     from .local_config import DataConfig, ModelConfig, TrainConfig, EvalConfig
# else:
from .server_config import DataConfig, ModelConfig, TrainConfig, EvalConfig
