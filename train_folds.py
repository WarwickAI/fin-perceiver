import wandb

from datasets import load_dataset
from transformers import PerceiverTokenizer, PerceiverForSequenceClassification, \
    Trainer, TrainingArguments, DataCollatorWithPadding

from metrics import compute_metrics
from dataset import k_fold_split

def train_folds(dataset, n_folds=10):
    tokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length')

    labels = dataset.features['label'].names
    id2label = { id: label for id, label in enumerate(labels) }
    label2id = { label: id for id, label in enumerate(labels) }

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples['sentence'],
            truncate=True
        ),
        batched=True
    )

    dataset_splits = k_fold_split(
        dataset=tokenized_dataset,
        n_splits=10,
        shuffle=True
    )

    default_training_args = {
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 4,
        'learning_rate': 2e-5,
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 2,
        'logging_strategy': 'steps',
        'logging_first_step': True,
        'logging_steps': 5,
        'report_to': 'wandb'
    }

    for current_fold, train, eval in enumerate(splits):
        print(f'Starting fold {current_fold}')

        model = PerceiverForSequenceClassification.from_pretrained(
            'deepmind/language-perceiver',
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )

        trainer = train_model(
            model=model,
            training_args=default_training_args,
            train=train,
            eval=eval
        )

        print(f'Finished training fold {current_fold}')

def train_model(model, training_args, train, eval):
    training_args = TrainingArguments(
        output_dir=f'fold_{current_fold}',
        **default_training_args
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    wandb.finish()

    return trainer

if __name__ == '__main__':
    wandb.login()

    financial_phrasebank = load_dataset(
        path='financial_phrasebank',
        name='sentences_50agree',
        split='train'
    )

    train_folds(financial_phrasebank, n_folds=10)