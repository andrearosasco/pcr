import abc
import importlib
import json
from abc import ABC, abstractmethod
import __main__


class ABCGetAttrMeta(abc.ABCMeta):
    def __getattribute__(self, item):
        try:
            return abc.ABCMeta.__getattribute__(self, item)
        except AttributeError:
            return __main__.Config.__getattribute__(__main__.Config, item)


class BaseConfig(metaclass=ABCGetAttrMeta):

    def __init__(self):
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
        if target is None:
            target = __main__.Config

        res = {}
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json']:
                attr = getattr(target, k)
                if type(attr) == type:
                    res[k] = __main__.Config.to_dict(attr)
                else:
                    res[k] = attr
        return res

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)
