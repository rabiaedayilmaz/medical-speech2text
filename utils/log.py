import logging
import logging.handlers as handlers
import os
import warnings


LOG_FILE_PATH = f'{os.environ["PYTHONPATH"]}/logs/server.log'

# format log
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# log for each day
log_handler = handlers.TimedRotatingFileHandler(LOG_FILE_PATH, when='midnight')
log_handler.setFormatter(formatter)

# print to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.addHandler(log_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("logger")