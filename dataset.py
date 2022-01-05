from datasets import load_dataset
from sklearn.model_selection import KFold

def get_split(split):
    train_idx, eval_idx = split

    return dataset.select(train_idx), \
        dataset.select(eval_idx)

def k_fold_split(n_splits, dataset, shuffle=True):
    fold = KFold(n_splits=n_splits, shuffle=True)

    return map(get_split, fold.split(dataset))