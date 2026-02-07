import logging
import time
import os


def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)

    log_name = "{}.log".format(time.strftime("%Y-%m-%d (%H:%M)"))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(final_log_file)
    console_handler = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
