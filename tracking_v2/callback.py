import inspect
from typing import Callable, Dict, List

__all__ = ['before_one', 'after_one', 'before_many', 'after_many', 'after_update', 'after_estimate', 'Runner',
           'execute_user_callbacks']



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


def target_cached(method):
    method.runner_callback_tag = 'target_cached'
    return method


def tracks_estimated(method):
    method.runner_callback_tag = 'tracks_estimated'
    return method


def associated_to_track(method):
    method.runner_callback_tag = 'associated_to_track'
    return method


def initialized_new_track(method):
    method.runner_callback_tag = 'initialized_new_track'
    return method


def measurement_not_associated(method):
    method.runner_callback_tag = 'measurement_not_associated'
    return method


def track_not_updated(method):
    method.runner_callback_tag = 'track_not_updated'
    return method

def mht_decisions(method):
    method.runner_callback_tag = 'mht_decisions'
    return method


# associates a given object ID (first-level key) with a mapping from stage name
# (second-level key) to a list of methods associated with that stage
CALLBACK_CACHE: Dict[int, Dict[str, List[Callable]]] = {}


def execute_callback(callback_object, stage, *args):
    """Execute all methods of the callback object decorated with the given stage.

    :param callback_object: Object whose methods are to be called during the execution of a tracking algorithm.
    :param stage: Stage name. If defined by the tracking algorithm, matching callback methods are executed.
    """
    global CALLBACK_CACHE
    
    if id(callback_object) not in CALLBACK_CACHE:
        CALLBACK_CACHE[id(callback_object)] = {}
    
    cached_callback_stages = CALLBACK_CACHE[id(callback_object)]
    
    if stage not in cached_callback_stages:
        callbacks = []
        
        for name, member in inspect.getmembers(callback_object, inspect.ismethod):
            if hasattr(member, 'runner_callback_tag') and member.runner_callback_tag == stage:
                callbacks.append(member)

        cached_callback_stages[stage] = callbacks
    
    for callback in cached_callback_stages[stage]:
        callback(*args)
