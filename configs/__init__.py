import importlib
import sys
from pathlib import Path

module = importlib.import_module(Path(sys.argv[0]).stem)

DataConfig = module.DataConfig
ModelConfig = module.ModelConfig
TrainConfig = module.TrainConfig
EvalConfig = module.EvalConfig
