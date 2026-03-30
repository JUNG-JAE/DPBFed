import inspect
from logging import handlers
import logging

def variable_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

class Logger:
    def __init__(self, log_file_name):
        self.log_formatter = logging.Formatter('%(asctime)s,%(message)s')
        self.handler = handlers.TimedRotatingFileHandler(filename='logs/' + str(log_file_name), when='midnight',
                                                         interval=1, encoding='utf-8')
        self.handler.setFormatter(self.log_formatter)
        self.handler.suffix = "%Y%m%d"

        # logger set
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def get_logger(self):
        return self.logger

    def log(self, message):
        print(message)
        self.logger.info(str(message))
        for hdlr in self.logger.handlers[:]:
            self.logger.removeHandler(hdlr)