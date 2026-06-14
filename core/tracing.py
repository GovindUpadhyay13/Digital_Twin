import time
import logging

logger = logging.getLogger("digital_twin.trace")

def traced(node_name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"[TRACE] {node_name} took {duration_ms:.1f}ms")
            return result
        return wrapper
    return decorator
