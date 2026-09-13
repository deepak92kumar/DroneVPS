import math

def is_finite_number(x):
    """Return True if x is a real number (not NaN/Inf)."""
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))

def clamp(val, min_val, max_val):
    """Clamp val between min_val and max_val."""
    return max(min(val, max_val), min_val)

def timestamp_fresh(ts, now, max_age=0.2):
    """Return True if timestamp ts is within max_age seconds of now.
    Used for VIO data freshness checks.
    """
    return (now - ts) <= max_age
