import inspect
from typing import Dict, List


__all__ = ['before_one', 'after_one', 'before_many', 'after_many', 'after_update', 'after_estimate', 'Runner']



def before_one(method):
    method.runner_callback_tag = 'before_one'
    return method


def after_one(method):
    method.runner_callback_tag = 'after_one'
    return method


def before_many(method):
    method.runner_callback_tag = 'before_many'
    return method


def after_many(method):
    method.runner_callback_tag = 'after_many'
    return method


def after_initialize(method):
    method.runner_callback_tag = 'after_initialize'
    return method


def after_predict(method):
    method.runner_callback_tag = 'after_predict'
    return method


def after_update(method):
    method.runner_callback_tag = 'after_update'
    return method


def after_estimate(method):
    method.runner_callback_tag = 'after_estimate'
    return method




class Runner:
    _callbacks: Dict = None


    def _execute_user_callbacks(self, stage, *args):
        if self._callbacks is None:
            self._callbacks = {}
        
        if stage not in self._callbacks:
            callbacks = []
            for name, member in inspect.getmembers(self, inspect.ismethod):
                if hasattr(member, 'runner_callback_tag') and member.runner_callback_tag == stage:
                    callbacks.append(member)
            self._callbacks[stage] = callbacks
        
        for callback in self._callbacks[stage]:
            callback(*args)

    def run_one(self, n: int, T: float = 1):
        raise Exception(f"run_one() method not implemented in {self.__class__.__name__}")

    def run_many(self, m: int, n: int, T: float = 1, seeds: List[int]=None):
        raise Exception(f"run_many() method not implemented in {self.__class__.__name__}")
