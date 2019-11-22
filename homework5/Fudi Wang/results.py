# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:13:52 2019

Results of A5 
"""

Loading data...
The `device` argument should be set by using `torch.device` or passing a string as an argument. This behavior will be deprecated soon and currently defaults to cpu.
The `device` argument should be set by using `torch.device` or passing a string as an argument. This behavior will be deprecated soon and currently defaults to cpu.

Parameters:
        BATCH_SIZE=64
        CLASS_NUM=2
        CUDA=False
        DEVICE=-1
        DROPOUT=0.5
        EARLY_STOP=1000
        EMBED_DIM=128
        EMBED_NUM=21108
        EPOCHS=32
        KERNEL_NUM=100
        KERNEL_SIZES=[3, 4, 5]
        LOG_INTERVAL=1
        LR=0.001
        MAX_NORM=3.0
        PREDICT=None
        SAVE_BEST=True
        SAVE_DIR=snapshot\2019-11-21_15-22-51
        SAVE_INTERVAL=500
        SHUFFLE=False
        SNAPSHOT=None
        STATIC=False
        TEST=False
        TEST_INTERVAL=100

Batch[50] - loss: 0.651913  acc: 57.8125%(37/64)
Batch[100] - loss: 0.717493  acc: 54.6875%(35/64)

Evaluation - loss: 0.000619 acc:60.4128%(644/1066) 
Batch[150] - loss: 0.647925  acc: 58.3333%(35/60)
Batch[200] - loss: 0.496167  acc: 81.2500%(52/64)

Evaluation - loss: 0.000595 acc:62.3827%(665/1066) 
Batch[250] - loss: 0.501821  acc: 78.1250%(50/64)
Batch[300] - loss: 0.509227  acc: 70.0000%(42/60)

Evaluation - loss: 0.000566 acc:65.5722%(699/1066) 
Batch[350] - loss: 0.276103  acc: 92.1875%(59/64)
Batch[400] - loss: 0.323175  acc: 85.9375%(55/64)

Evaluation - loss: 0.000554 acc:68.5741%(731/1066) 
Batch[450] - loss: 0.311763  acc: 93.3333%(56/60)
Batch[500] - loss: 0.081068  acc: 100.0000%(64/64)

Evaluation - loss: 0.000563 acc:69.7936%(744/1066) 
Batch[550] - loss: 0.112142  acc: 100.0000%(64/64)
Batch[600] - loss: 0.118519  acc: 100.0000%(60/60)

Evaluation - loss: 0.000580 acc:69.4184%(740/1066) 
Batch[650] - loss: 0.032185  acc: 100.0000%(64/64)
Batch[700] - loss: 0.045586  acc: 100.0000%(64/64)

Evaluation - loss: 0.000602 acc:70.3565%(750/1066) 
Batch[750] - loss: 0.039748  acc: 100.0000%(60/60)
Batch[800] - loss: 0.016468  acc: 100.0000%(64/64)

Evaluation - loss: 0.000627 acc:71.7636%(765/1066) 
Batch[850] - loss: 0.013309  acc: 100.0000%(64/64)
Batch[900] - loss: 0.009398  acc: 100.0000%(60/60)

Evaluation - loss: 0.000642 acc:72.8893%(777/1066) 

