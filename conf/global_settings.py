import os
from datetime import datetime


# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
# total training epoches
EPOCH = 1000
MILESTONES = [i for i in range(1, 1000, 10)]  # 每10个epoch调整学习率
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
# tensorboard log dir
LOG_DIR = 'runs'
# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
