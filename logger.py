import logging
import logging.config
import os

class Logger:          
    def logger_init():
        try:
            logging.config.fileConfig('logging.conf')
            logger = logging.getLogger('Logger')
            return logger
        except Exception as e:
            return f'Logging error {e}'