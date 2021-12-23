import importlib
import sys
from pathlib import Path

# If you want to execute an arbitrary file and import
# a specific configuration, add these lines before the
# configuration import:
# import sys
# sys.argv[0] = '[name of the config file without extension]'

module = importlib.import_module(f'configs.{Path(sys.argv[0]).stem}')

DataConfig = module.DataConfig
ModelConfig = module.ModelConfig
TrainConfig = module.TrainConfig
EvalConfig = module.EvalConfig
