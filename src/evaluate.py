from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np
import pandas as pd

# Public function called from app.py to evaluate the prompt
def evaluate_prompt(model_name, prompt):
    # Check if the model is fine-tuned-toxic-tweets
    if model_name == "fine-tuned-toxic-tweets":
        return eval_fine_tuned_toxic_tweets(model_name, prompt)
    else: # If not, use the pipeline function
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier(prompt)

# Private function to evaluate the prompt using the fine-tuned-toxic-tweets model
def eval_fine_tuned_toxic_tweets(model_name, prompt):
    # Load the model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained("sergey-hovhannisyan/fine-tuned-toxic-tweets")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encoded_text = tokenizer(prompt, truncation=True, padding=True, return_tensors='pt')

    with torch.no_grad():
        output = model(**encoded_text)

    # Get the labels and scores
    labels = np.array(["toxic", "severe", "obscene", "threat", "insult", "identity hate"])
    scores = torch.sigmoid(output.logits)*100
    scores = scores.numpy()

    # Sort the scores in descending order
    sort_idx = np.flip(np.argsort(scores))
    labels = labels[sort_idx]
    scores = scores[0][sort_idx]
    
    # Return the labels and scores as a dataframe
    return pd.DataFrame({"Label": labels[0].tolist(), "Score": scores[0].tolist()}).set_index("Label")

# List of models to choose from
def model_list():
    sentiment_models = [
        "fine-tuned-toxic-tweets",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "bert-base-uncased",
        "albert-base-v2"]
    return sentiment_models