from src.logger import Logger
from src.exception import CustomException
import sys

_logger_obj = Logger("main")
logger = _logger_obj.get_logger()

logger.info("Logger working for Project 5!")

try:
    x = 1 / 0
except Exception as e:
    raise CustomException(e, sys)