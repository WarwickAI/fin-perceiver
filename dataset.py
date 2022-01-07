from sklearn.model_selection import KFold

def k_fold_split(n_splits, dataset, shuffle=True):
    fold = KFold(n_splits=n_splits, shuffle=True)

    return map(
        lambda split: (
            dataset.select(split[0]),
            dataset.select(split[1])
        ),
        fold.split(dataset)
    )