import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings('ignore')
from A4 import mixup as a4
# ======================================================================================================================
# Data preprocessing
train,test = a4.preProcessing()
# # Task A4
acc_train = a4.train(train) 
acc_test = a4.test(test)
# # ====================================================================================================================
print('A4: Train score:{};Test score:{};'.format(acc_train, acc_test))