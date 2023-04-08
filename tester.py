import pathvision.core as pathvision
from pathvision.core.logger import logger as LOGGER

def tester():
    LOGGER.debug("Debug message")
    LOGGER.info("Info message")
    LOGGER.warning("Warning message")
    LOGGER.error("Error message")
    LOGGER.critical("Critical message")

if __name__ == "__main__":
    tester()