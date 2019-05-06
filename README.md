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

=> [INFO] Load and parse raw data from data/criteo.tr.r100.gbdt0.ffm ... <=
 => [INFO] Load and parse raw data from data/criteo.va.r100.gbdt0.ffm ... <=
 => [INFO] Process 99% in Epoch 1: [Train] log-loss: 0.48147 <=
 => [INFO] STEP 1, [Val] val_loss: 0.48261, val_auc: 0.768 <=
 => [INFO] Process 86% in Epoch 2: [Train] log-loss: 0.40697 <=