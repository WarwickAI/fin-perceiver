# FINPerceiver
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-warwickai%2Ffin--perceiver-yellow)](https://huggingface.co/warwickai/fin-perceiver)

FINPerceiver is a fine-tuned Perceiver IO model for financial sentiment analysis.

We achieved the following results with 10-fold cross validation.

```
eval/accuracy  0.8624 (stdev 0.01922)
eval/f1        0.8416 (stdev 0.03738)
eval/loss      0.4314 (stdev 0.05295)
eval/precision 0.8438 (stdev 0.02938)
eval/recall    0.8415 (stdev 0.04458)
```

The hyperparameters used are as follows.

```
per_device_train_batch_size  16
per_device_eval_batch_size   16
num_train_epochs             4
learning_rate                2e-5
```

## Training
Create W&B API token, W&B/HF CLI login, ... (TBD)

```
pip3 install -r requirements.txt
WANDB_PROJECT=fin_perceiver python train_folds.py
```

## Datasets
This model was trained on the Financial PhraseBank (>= 50% agreement) from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts)
