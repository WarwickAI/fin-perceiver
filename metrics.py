from datasets import load_metric

def compute_metrics(eval_pred):
    recall_metric = load_metric('recall')
    f1_metric = load_metric('f1')
    accuracy_metric = load_metric('accuracy')
    precision_metric = load_metric('precision')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    recall = recall_metric.compute(
            predictions=predictions,
            references=labels,
            average='macro'
    )['recall']

    f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average='macro'
    )['f1']

    accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels
    )['accuracy']

    precision = precision_metric.compute(
            predictions=predictions,
            references=labels,
            average='macro'
    )['precision']

    return {
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision
    }