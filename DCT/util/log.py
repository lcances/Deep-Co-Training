import logging
import numpy as np

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from logging import LogRecord
from typing import Optional

def log_flat(array) -> str:
    if isinstance(array, (int, float, bool, str)):
        return array
    return "[" + ", ".join(map(str, np.asarray(array).reshape(-1))) + "]"

"""
ColoredLog found : https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
"""
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        formatted_msg = super().format(record)

        return COLOR_SEQ % (30 + COLORS[record.levelname]) + formatted_msg + RESET_SEQ


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            "()": ColoredFormatter,
            'format': '%(levelname)s --- %(module)s.%(funcName)s >>> %(message)s'
        },
    },

    'handlers': {
        'default': {
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },

    'loggers': {
        '': {
            "handlers": ["default"],
            'level': 'DEBUG',
        },
    }
}