from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

# GPU CHECK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Set load
dataset = load_dataset("imdb")

# Step 3: Preprocess the Data with Tokenizer
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function to prepare inputs for BERT
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization to the datasets
train_dataset = dataset["train"].map(tokenize, batched=True)
test_dataset = dataset["test"].map(tokenize, batched=True)

# Remove text column and prepare data for PyTorch
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Convert datasets to PyTorch format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 4: Load Pretrained BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Step 5: Define Metrics for Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Move logits to CPU and convert to NumPy
    predictions = torch.argmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
    labels = torch.tensor(labels).detach().cpu().numpy()  # Ensure labels are on CPU
    accuracy = accuracy_score(labels, predictions)  # Compute accuracy
    return {"accuracy": accuracy}

# Step 6: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the model checkpoints
    evaluation_strategy="epoch",     # Evaluate after each epoch
    save_strategy="epoch",           # Save model at each epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    num_train_epochs=2,              # Number of epochs
    weight_decay=0.01,               # Weight decay (L2 regularization)
    logging_steps=10,                # Log every 10 steps
    load_best_model_at_end=True,     # Load best model at the end of training
    fp16=True,                       # Use mixed precision for faster training on GPU

    #for some reason the next line is needed due to a reporting system
    report_to="none"                 # Disable WandB logging
)

# Step 7: Initialize the Trainer with Model and Datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Use corrected function
)

# Step 8: Train the Model (GPU will be used automatically if available)
trainer.train()

# Step 9: Evaluate the Model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Step 10: Predict Sentiment on New Movie Reviews
def predict(texts):
    # Prepare input data for prediction
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Ensure GPU usage

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return ["Positive" if pred == 1 else "Negative" for pred in predictions]

# Step 11: Example Sentiment Predictions
sample_reviews = [
    "The movie was fantastic! The characters were well-developed and the plot was engaging.",
    "I did not enjoy the film. It was too long and the story was boring.",
    "An amazing experience! One of the best movies I have ever seen.",
    "Terrible movie. Waste of time."
]

predictions = predict(sample_reviews)
for review, sentiment in zip(sample_reviews, predictions):
    print(f"Review: {review}\nSentiment: {sentiment}\n")
