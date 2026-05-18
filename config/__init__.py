try:
    from config.config import *
except ImportError:
    raise ImportError(
        "config/config.py not found.\n"
        "Copy config/config.template.py to config/config.py and fill in your AG_API_KEY."
    )
