from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def evaluate_prompt(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(prompt)

def model_list():
    sentiment_models = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "bert-base-uncased",
        "albert-base-v2"]
    return sentiment_models