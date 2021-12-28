# FINPerceiver

FINPerceiver is a fine-tuned Perceiver IO model for financial sentiment analysis.

We achieved the following results on a 20% random evaluation subset of the Financial PhraseBank.

```
eval/accuracy  0.8649
eval/f1        0.8515
eval/loss      0.396
eval/precision 0.8542
eval/recall    0.8502
```

TBD: 10-fold cross validation

## Installation
Create W&B API token, W&B/HF CLI login, ....

## Training
`WANDB_PROJECT=fin_perceiver python train.py`

## Datasets
This model was trained on the Financial PhraseBank (>= 50% agreement) from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts)
