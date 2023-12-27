# Copied from https://stackoverflow.com/a/1336640

import logging
import platform
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def add_coloring_to_emit_windows(fn):
    # add methods we need to the class
    def _out_handle(self):
        import ctypes
        return ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
    out_handle = property(_out_handle)

    def _set_color(self, code):
        import ctypes

        # Constants from the Windows API
        self.STD_OUTPUT_HANDLE = -11
        hdl = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
        ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code)

    setattr(logging.StreamHandler, '_set_color', _set_color)

    def new(*args):
        FOREGROUND_BLUE = 0x0001  # text color contains blue.
        FOREGROUND_GREEN = 0x0002  # text color contains green.
        FOREGROUND_RED = 0x0004  # text color contains red.
        FOREGROUND_INTENSITY = 0x0008  # text color is intensified.
        FOREGROUND_WHITE = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED
        # winbase.h
        # STD_INPUT_HANDLE = -10
        # STD_OUTPUT_HANDLE = -11
        # STD_ERROR_HANDLE = -12

        # wincon.h
        # FOREGROUND_BLACK = 0x0000
        FOREGROUND_BLUE = 0x0001
        FOREGROUND_GREEN = 0x0002
        # FOREGROUND_CYAN = 0x0003
        FOREGROUND_RED = 0x0004
        FOREGROUND_MAGENTA = 0x0005
        FOREGROUND_YELLOW = 0x0006
        # FOREGROUND_GREY = 0x0007
        FOREGROUND_INTENSITY = 0x0008  # foreground color is intensified.

        # BACKGROUND_BLACK = 0x0000
        # BACKGROUND_BLUE = 0x0010
        # BACKGROUND_GREEN = 0x0020
        # BACKGROUND_CYAN = 0x0030
        # BACKGROUND_RED = 0x0040
        # BACKGROUND_MAGENTA = 0x0050
        BACKGROUND_YELLOW = 0x0060
        # BACKGROUND_GREY = 0x0070
        BACKGROUND_INTENSITY = 0x0080  # background color is intensified.

        levelno = args[1].levelno
        if (levelno >= 50):
            color = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
        elif (levelno >= 40):
            color = FOREGROUND_RED | FOREGROUND_INTENSITY
        elif (levelno >= 30):
            color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
        elif (levelno >= 20):
            color = FOREGROUND_GREEN
        elif (levelno >= 10):
            color = FOREGROUND_MAGENTA
        else:
            color = FOREGROUND_WHITE
        args[0]._set_color(color)

        ret = fn(*args)
        args[0]._set_color(FOREGROUND_WHITE)
        # print "after"
        return ret
    return new


def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[1].levelno if not isinstance(args[1], Exception) else 50
        msg = str(args[1]) if isinstance(args[1], Exception) else str(args[1].msg)

        if (levelno >= 50):
            color = '\x1b[31m'  # red
        elif (levelno >= 40):
            color = '\x1b[31m'  # red
        elif (levelno >= 30):
            color = '\x1b[33m'  # yellow
        elif (levelno >= 20):
            color = '\x1b[32m'  # green
        elif (levelno >= 10):
            color = '\x1b[35m'  # pink
        else:
            color = '\x1b[0m'  # normal

        args[1].msg = color + msg + '\x1b[0m'  # normal

        return fn(*args)
    
    return new


if platform.system() == 'Windows':
    # Windows does not support ANSI escapes and we are using API calls to set the console color
    logging.StreamHandler.emit = add_coloring_to_emit_windows(logging.StreamHandler.emit)
else:
    # all non-Windows platforms are supporting ANSI escapes so we use them
    logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

def get_loglevel():
    log_level = "INFO"
    if "LOG_LEVEL" in os.environ:
        log_level = os.environ["LOG_LEVEL"]
    if log_level not in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]:
        raise Exception(f"Invalid log_level={log_level}")
    log_level = getattr(logging, log_level.upper())
    print(f"get_loglevel: log_level={log_level}")
    return log_level

global_log_level = None

def configure_loglevel():
    global global_log_level
    global_log_level =get_loglevel()

def getLogger(name=None, level=None):
    global global_log_level
    if not name:
        name = __name__
    logger = logging.getLogger(name)
    if not global_log_level:
        configure_loglevel()
    if not level:
        level = global_log_level
    logger.setLevel(level)
    return logger

