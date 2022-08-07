from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing
###gpu:查看gpu端口
### Logger:日记类
def gpu(index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # 指定GPU的id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.total / 1024 ** 2)  # 总的显存大小（float）
    print(meminfo.used / 1024 ** 2)
    print(meminfo.free / 1024 ** 2)

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
