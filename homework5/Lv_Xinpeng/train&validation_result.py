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
        EPOCHS=256
        KERNEL_NUM=100
        KERNEL_SIZES=[3, 4, 5]
        LOG_INTERVAL=1
        LR=0.001
        MAX_NORM=3.0
        PREDICT=None
        SAVE_BEST=True
        SAVE_DIR=snapshot\2019-11-20_10-42-41
        SAVE_INTERVAL=500
        SHUFFLE=False
        SNAPSHOT=None
        STATIC=False
        TEST=False
        TEST_INTERVAL=100

Batch[100] - loss: 0.8018  acc: 46.8750%(30/64)
Evaluation - loss: 0.6594  acc: 60.0375%(640/1066) 

Batch[200] - loss: 0.5518  acc: 67.1875%(43/64)
Evaluation - loss: 0.6379  acc: 63.2270%(674/1066) 

Batch[300] - loss: 0.5749  acc: 68.3333%(41/60)
Evaluation - loss: 0.6135  acc: 67.0732%(715/1066) 

Batch[400] - loss: 0.6540  acc: 65.6250%(42/64)
Evaluation - loss: 0.5977  acc: 67.6360%(721/1066) 

Batch[500] - loss: 0.5136  acc: 75.0000%(48/64)
Evaluation - loss: 0.6139  acc: 66.4165%(708/1066) 

Batch[600] - loss: 0.5308  acc: 76.6667%(46/60)
Evaluation - loss: 0.5781  acc: 71.1069%(758/1066) 

Batch[700] - loss: 0.4598  acc: 81.2500%(52/64)
Evaluation - loss: 0.5951  acc: 71.2946%(760/1066) 

Batch[800] - loss: 0.1887  acc: 95.3125%(61/64))
Evaluation - loss: 0.5903  acc: 72.0450%(768/1066) 

Batch[900] - loss: 0.2430  acc: 90.0000%(54/60))
Evaluation - loss: 0.6627  acc: 68.1989%(727/1066) 

Batch[1000] - loss: 0.0991  acc: 96.8750%(62/64)
Evaluation - loss: 0.6286  acc: 73.2645%(781/1066) 

Batch[1100] - loss: 0.0941  acc: 96.8750%(62/64))
Evaluation - loss: 0.6616  acc: 72.0450%(768/1066) 

Batch[1200] - loss: 0.1574  acc: 93.3333%(56/60))
Evaluation - loss: 0.6724  acc: 72.7955%(776/1066) 

Batch[1300] - loss: 0.1159  acc: 93.7500%(60/64))
Evaluation - loss: 0.7185  acc: 73.4522%(783/1066) 

Batch[1400] - loss: 0.0557  acc: 98.4375%(63/64))
Evaluation - loss: 0.7665  acc: 71.8574%(766/1066) 

Batch[1500] - loss: 0.0686  acc: 96.6667%(58/60))
Evaluation - loss: 0.7975  acc: 73.4522%(783/1066) 

Batch[1600] - loss: 0.0253  acc: 98.4375%(63/64))
Evaluation - loss: 0.8227  acc: 73.0769%(779/1066) 

Batch[1700] - loss: 0.1005  acc: 96.8750%(62/64))
Evaluation - loss: 0.8638  acc: 72.7955%(776/1066) 

Batch[1800] - loss: 0.0099  acc: 100.0000%(60/60)
Evaluation - loss: 0.9018  acc: 73.7336%(786/1066) 

Batch[1900] - loss: 0.0690  acc: 98.4375%(63/64))
Evaluation - loss: 0.9521  acc: 74.5779%(795/1066) 

Batch[2000] - loss: 0.0210  acc: 100.0000%(64/64)
Evaluation - loss: 0.9894  acc: 74.2964%(792/1066) 
"""
