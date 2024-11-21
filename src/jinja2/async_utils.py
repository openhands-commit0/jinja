import inspect
import typing as t
from functools import WRAPPER_ASSIGNMENTS
from functools import wraps
from .utils import _PassArg
from .utils import pass_eval_context
V = t.TypeVar('V')
_common_primitives = {int, float, bool, str, list, dict, tuple, type(None)}

def async_variant(normal_func: t.Callable[..., V]) -> t.Callable[..., t.Awaitable[V]]:
    """Returns an async variant of a sync function. This is useful to provide
    async versions of filters and tests.

    Example::

        def foo(x):
            return x

        async_foo = async_variant(foo)

    When called, the original function will be run in a thread executor.
    """
    is_async = inspect.iscoroutinefunction(normal_func)
    is_filter = getattr(normal_func, '_jinja_filter', False)
    is_test = getattr(normal_func, '_jinja_test', False)
    is_pass_arg = getattr(normal_func, '_pass_arg', None)
    is_pass_eval_context = getattr(normal_func, '_pass_eval_context', False)

    if is_async:
        return normal_func

    async def async_func(*args: t.Any, **kwargs: t.Any) -> V:
        if is_pass_arg is not None:
            kwargs.pop(is_pass_arg.value, None)
        if is_pass_eval_context:
            args = args[1:]
        return await normal_func(*args, **kwargs)

    # Properly rewrap the function to carry over all the original attributes
    # and metadata of the original function
    async_func = wraps(normal_func, assigned=WRAPPER_ASSIGNMENTS)(async_func)
    if is_filter:
        async_func._jinja_filter = True  # type: ignore
    if is_test:
        async_func._jinja_test = True  # type: ignore
    if is_pass_arg is not None:
        async_func._pass_arg = is_pass_arg  # type: ignore
    if is_pass_eval_context:
        async_func._pass_eval_context = True  # type: ignore

    return async_func

async def auto_aiter(value: t.Any) -> t.AsyncIterator[t.Any]:
    """Convert an iterable into an async iterable. This is useful to
    iterate over sync iterables in async contexts.

    Example::

        async for item in auto_aiter([1, 2, 3]):
            ...
    """
    if hasattr(value, '__aiter__'):
        async for item in value:
            yield item
    else:
        for item in value:
            yield item