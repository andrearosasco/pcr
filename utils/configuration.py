import abc
import importlib
import json
from abc import abstractmethod

class ABCGetAttrMeta(abc.ABCMeta):
    cfg = None

    def __getattribute__(self, item):
        try:
            return abc.ABCMeta.__getattribute__(self, item)
        except AttributeError:
            return ABCGetAttrMeta.cfg.__getattribute__(item)


class BaseConfig(metaclass=ABCGetAttrMeta):

    def __init__(self):
        ABCGetAttrMeta.cfg = self

        mod = importlib.import_module(self.run)
        getattr(mod, 'main')()

    @property
    @abstractmethod
    def run(self) -> str:
        """" A string pointing to the script to run"""
        pass

    @classmethod
    def to_json(cls):
        return json.dumps(cls.to_dict())

    @classmethod
    def to_dict(cls, target=None):
        from configs import Config

        if target is None:
            target = Config

        res = {}
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json']:
                attr = getattr(target, k)
                if type(attr) == type:
                    res[k] = Config.to_dict(attr)
                else:
                    res[k] = attr
        return res

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)
