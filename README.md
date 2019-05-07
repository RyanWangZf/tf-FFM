# tf-FFM
Tensorflow based field-aware factorization machine, FFM.

## Usage

```python
from config import config
from dataset import Dataset
from ffm import FFM

ffm = FFM(config)
tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
va_filename = "data/criteo.va.r100.gbdt0.ffm"
dataset = Dataset(tr_filename, va_filename, config.batch_size, config.shuffle)
ffm.train(dataset)
```

## Result
=> Config Settings <=  
Learning rate: 0.001  
L2 normalization: 5e-05  
Batch size: 1024  
Embedding size: 4  
Total epoch: 20  
=> Config Settings <=  
=> [INFO] Load and parse raw data from data/criteo.tr.r100.gbdt0.ffm ... <=  
=> [INFO] Load and parse raw data from data/criteo.va.r100.gbdt0.ffm ... <=  
=> [INFO] Process 100% in Epoch 1: [Train] log-loss: 0.46042 one iter 0.2 sec <=   
=> [INFO] STEP 1, [Val] val_loss: 0.47963, val_auc: 0.770 one epoch in 70.9 sec <=  
=> [INFO] Process 100% in Epoch 2: [Train] log-loss: 0.45366 one iter 0.2 sec <=   
=> [INFO] STEP 2, [Val] val_loss: 0.47581, val_auc: 0.775 one epoch in 71.0 sec <=  
=> [INFO] Process 100% in Epoch 3: [Train] log-loss: 0.42570 one iter 0.2 sec <=   
=> [INFO] STEP 3, [Val] val_loss: 0.47484, val_auc: 0.776 one epoch in 71.0 sec <=  
=> [INFO] Process 100% in Epoch 4: [Train] log-loss: 0.45457 one iter 0.2 sec <=   
=> [INFO] STEP 4, [Val] val_loss: 0.47581, val_auc: 0.775 one epoch in 71.2 sec <=  
