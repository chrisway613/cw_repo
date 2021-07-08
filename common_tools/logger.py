"""在开发项目过程中使用的日志记录器，使用内置的logging模块提供的API，
可以很方便地在将日志信息输出到控制台的同时记录到文件，这样就能实现即时查看以及回溯的需求。"""

import os
import logging


class Logger(logging.Logger):
    """日志记录模块，继承logging.Logger。
           使用该模块可以使得日志在输出到控制台的同时也记录到文件，从而满足即时查看及后续回溯的需求。"""

    # 日志文件格式，'.log'或'.txt'
    fmts = ('log', 'txt')

    def __init__(self, path, level=logging.INFO):
        """
        :param path: 日志输出文件的路径；
        :param level: 日志等级，参考logging模块
        """

        # 设置logger名称
        # 当路径最后是'/'时，logger_name会是''，这时候需要手工指定一个log文件名词
        logger_name = os.path.basename(path)
        if not logger_name:
            logger_name = 'root'

        parts = logger_name.split('.')
        if len(parts) == 2:
            logger_name, fmt = parts
        else:
            logger_name = parts[0]
            fmt = self.fmts[0]

        assert fmt in ('log', 'txt'), f"log format should be '.log' or '.txt', got {fmt}"
        super().__init__(logger_name, level)

        # 日志文件存放的目录
        log_dir = os.path.dirname(path)
        # 目录不存在则创建(包括中间路径的所有目录)
        # 若存在则无影响(设置了exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # 日志文件存放路径
        self.log_path = os.path.join(log_dir, f"{logger_name}.{fmt}")

        self._init_logger()

    def _init_logger(self):
        # 配置文件句柄(输出到文件)
        file_handler = logging.FileHandler(self.log_path, mode='w')
        file_handler.setLevel(self.level)
        # 设置日志内容的格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置控制台句柄(输出到屏幕)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)

        # 添加句柄
        self.addHandler(file_handler)
        self.addHandler(console_handler)
