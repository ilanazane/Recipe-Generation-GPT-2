import torch 
from transformers import(
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

# add a pad token to the tokenizer 
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# load the dataset 
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer, file_path=file_path, block_size=block_size
    )

train_dataset = load_dataset("data/cleaned_recipes.csv", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# set the training arguments 
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10000,
    save_total_limit=2,
)

# create the trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# train the model 
trainer.train()

# save the model and tokenizer 
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("model and tokenizer saved")

