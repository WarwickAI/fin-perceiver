import wandb

from metrics import compute_metrics

from transformers import PerceiverTokenizer, PerceiverForSequenceClassification, \
    Trainer, TrainingArguments

train_dataset, eval_dataset = load_dataset(
    path='financial_phrasebank',
    name='sentences_50agree',
    split=['train[:80%]', 'train[80%:100%]'])

tokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

labels = train_dataset.features['label'].names
id2label = { id: label for id, label in enumerate(labels) }
label2id = { label: id for id, label in enumerate(labels) }

wandb.login()

model = PerceiverForSequenceClassification.from_pretrained(
    'deepmind/language-perceiver',
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=3.45e-6,
    evaluation_strategy='steps',
    eval_steps=100,
    save_strategy='epoch',
    save_total_limit=2,
    output_dir='fin-perceiver',
    logging_strategy='steps',
    logging_first_step=True,
    logging_steps=5,
    report_to='wandb'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
wandb.finish()