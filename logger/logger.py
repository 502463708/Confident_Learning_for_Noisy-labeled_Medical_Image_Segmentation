import os


class Logger(object):
    def __init__(self, log_saving_dir, log_filename='log.txt'):
        assert os.path.isdir(log_saving_dir)
        log_saving_path = os.path.join(log_saving_dir, log_filename)
        self.logger = open(log_saving_path, "w", encoding="utf-8")

        return

    def write(self, message):
        self.logger.write(message)
        self.logger.write('\n')

        return

    def write_and_print(self, message):
        self.logger.write(message)
        self.logger.write('\n')

        print(message)

        return

    def flush(self):
        self.logger.flush()

        return
