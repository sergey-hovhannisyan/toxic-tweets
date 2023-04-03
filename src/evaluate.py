from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def evaluate_prompt(model_name, prompt):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
    return classifier(prompt)

def model_list():
     return ["bert", "robert", "albert"]
