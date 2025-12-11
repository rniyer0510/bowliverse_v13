import sys
import logging

# --------------------------------------------------------
# Create a unified logger for all Bowliverse modules
# --------------------------------------------------------
LOGGER_NAME = "bowliverse"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)     # Allow all levels

# If no handlers exist, add one (avoid duplicate logs)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False  # Prevent duplicate uvicorn logs


# --------------------------------------------------------
# Backward compatible simple print interface
# --------------------------------------------------------
def log(msg):
    """
    Backward compatible print.
    Old code using log('...') will still work.
    """
    logger.info(msg)


# Explicit level helpers
def debug(msg):
    logger.debug(msg)

def info(msg):
    logger.info(msg)

def warn(msg):
    logger.warning(msg)

def error(msg):
    logger.error(msg)

