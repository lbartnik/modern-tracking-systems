import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
import inspect
from typing import Dict, List

from ..np import as_column



__all__ = ['before_one', 'after_one', 'before_many', 'after_many', 'after_update', 'Runner']



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
