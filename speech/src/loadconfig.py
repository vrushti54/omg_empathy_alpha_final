import os

def load():
    """Return absolute path to config.ini located next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "config.ini")
