import functools


def entrypoint(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Executing {func.__name__} with arguments {args} and keyword arguments {kwargs}"
        )
        result = func(*args, **kwargs)
        print(f"Finished executing {func.__name__}")
        return result

    wrapper.__wrapped__ = func
    wrapper.__qualname__ = "entrypoint"
    return wrapper
