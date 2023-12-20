#import os
from datetime import datetime

# mean and standard deviation of data
FMST_TRAIN_MEAN = (0.2861,)
FMST_TRAIN_STD = (0.3528,)

# hyper-parameters for training
EPOCH = 90
MILESTONES = [20, 40, 60]
# save weights per epoch
EPOCH_SAVE = 20

# path for saved weight
CHECKPOIN_DIR = './checkpoint'
# current time
CURR_TIME = datetime.now().isoformat()

# path for tensorboard log
LOG_DIR = './runs/fmst'
