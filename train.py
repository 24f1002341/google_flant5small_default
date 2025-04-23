from datasets import load_dataset
from transformers import T5Tokenizer,T5ForConditionalGeneration
import transformers



#load dataset
dataset= load_dataset('json',data_files='data.json')

dataset=dataset["train"].train_test_split(test_size=0.1)

#tokenise dataset

tokenizer=T5Tokenizer.from_pretrained('google/flan-t5-small')

def tokenize(batch):
    input_enc=tokenizer(batch["input"],padding="max_length",truncation=True,max_length=128)
    output_enc=tokenizer(batch["output"],padding="max_length",truncation=True,max_length=64)


    labels = output_enc["input_ids"]
    labels = [[(token if token != tokenizer.pad_token_id else -100) for token in label_seq] for label_seq in labels]

    input_enc["labels"] = labels
    return input_enc
tokenized_dataset = dataset.map(tokenize, batched=True)

#fine tuning

model=T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
args=transformers.TrainingArguments(
    output_dir="./flan-t5-small-finetuned",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    remove_unused_columns=False,

)

print('machine is learning')

trainer=transformers.Trainer(model=model,
                args=args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset['test'])

trainer.train()

model.save_pretrained("./flan-t5-small-finetuned")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")  # Or wherever your base tokenizer came from
tokenizer.save_pretrained("./flan-t5-small-finetuned")







