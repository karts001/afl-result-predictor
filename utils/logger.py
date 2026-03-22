import logging
import sys

sys.stdout.reconfigure(encoding='utf-8')

# create the logger
logger = logging.getLogger("scraper_logger")
logger.setLevel(level=logging.DEBUG)#

# create a file handler
file_handler = logging.FileHandler(filename="scraper.log")
file_handler.setLevel(level=logging.DEBUG)

# create a stream handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level=logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(console_handler)