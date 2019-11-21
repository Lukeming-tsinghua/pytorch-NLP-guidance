# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''
Parameters:
        BATCH_SIZE=64
        CLASS_NUM=2
        CUDA=False
        DEVICE=-1
        DROPOUT=0.5
        EARLY_STOP=1000
        EMBED_DIM=128
        EMBED_NUM=21114
        EPOCHS=256
        KERNEL_NUM=100
        KERNEL_SIZES=[3, 4, 5]
        LOG_INTERVAL=1
        LR=0.001
        MAX_NORM=3.0
        PREDICT=None
        SAVE_BEST=True
        SAVE_DIR=snapshot/2019-11-21_14-06-04
        SAVE_INTERVAL=500
        SHUFFLE=False
        SNAPSHOT=None
        STATIC=False
        TEST=False
        TEST_INTERVAL=100

Batch[100] - loss: 0.7846  acc: 51.5625%(33/64)
Evaluation - loss: 0.6423  acc: 63.9775%(682/1066) 
          
Batch[200] - loss: 0.5949  acc: 64.0625%(41/64)
Evaluation - loss: 0.6168  acc: 66.7917%(712/1066) 
          
Batch[300] - loss: 0.5443  acc: 75.0000%(45/60)
Evaluation - loss: 0.6234  acc: 64.2589%(685/1066) 
          
Batch[400] - loss: 0.5999  acc: 70.3125%(45/64)
Evaluation - loss: 0.5787  acc: 69.3246%(739/1066) 
          
Batch[500] - loss: 0.4536  acc: 79.6875%(51/64)
Evaluation - loss: 0.5776  acc: 70.3565%(750/1066) 

Batch[600] - loss: 0.4922  acc: 76.6667%(46/60)
Evaluation - loss: 0.5663  acc: 70.8255%(755/1066) 
          
Batch[700] - loss: 0.4060  acc: 81.2500%(52/64)
Evaluation - loss: 0.5563  acc: 72.4203%(772/1066) 
          
Batch[800] - loss: 0.2435  acc: 90.6250%(58/64)
Evaluation - loss: 0.5861  acc: 72.7955%(776/1066) 
          
Batch[900] - loss: 0.3256  acc: 90.0000%(54/60)
Evaluation - loss: 0.6109  acc: 72.3265%(771/1066) 
          
Batch[1000] - loss: 0.2341  acc: 92.1875%(59/64)
Evaluation - loss: 0.6317  acc: 73.4522%(783/1066) 
          
Batch[1100] - loss: 0.0905  acc: 95.3125%(61/64)
Evaluation - loss: 0.6575  acc: 72.9831%(778/1066) 
          
Batch[1200] - loss: 0.1857  acc: 95.0000%(57/60)
Evaluation - loss: 0.6705  acc: 74.5779%(795/1066) 
          
Batch[1300] - loss: 0.0665  acc: 98.4375%(63/64)
Evaluation - loss: 0.7078  acc: 74.7655%(797/1066) 

Batch[1400] - loss: 0.0377  acc: 98.4375%(63/64)
Evaluation - loss: 0.7430  acc: 74.3902%(793/1066) 
          
Batch[1500] - loss: 0.0828  acc: 98.3333%(59/60)
Evaluation - loss: 0.7898  acc: 74.9531%(799/1066) 
          
Batch[1600] - loss: 0.0710  acc: 98.4375%(63/64)
Evaluation - loss: 0.8229  acc: 74.5779%(795/1066) 
          
Batch[1700] - loss: 0.0411  acc: 98.4375%(63/64)
Evaluation - loss: 0.8605  acc: 73.8274%(787/1066) 
          
Batch[1800] - loss: 0.0087  acc: 100.0000%(60/60)
Evaluation - loss: 0.8957  acc: 74.4841%(794/1066) 
          
Batch[1900] - loss: 0.0274  acc: 98.4375%(63/64)
Evaluation - loss: 0.9058  acc: 74.5779%(795/1066) 
          
Batch[2000] - loss: 0.0719  acc: 96.8750%(62/64)
Evaluation - loss: 0.9478  acc: 74.2964%(792/1066) 
          
Batch[2100] - loss: 0.0153  acc: 100.0000%(60/60)
Evaluation - loss: 0.9977  acc: 73.8274%(787/1066) 
