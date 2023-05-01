---
title: Toxic Tweets
emoji: 😈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
license: bsd-3-clause
---
# _Toxic Tweets_
# _Objective_ 
The primary objective of this machine learning project in Natural Language Processing (NLP) is to accurately predict the toxicity level of tweets by utilizing fine-tuning techniques on a pre-trained language model. To achieve this, the project utilizes Docker containers for efficient deployment and management and Hugging Face Spaces and Transformers libraries for model building and training. The model is trained on a substantial dataset of labeled toxic tweets, allowing it to classify new tweets into five toxicity labels: "severe toxic," "obscene," "threat," "insult," and "identity hate." By integrating the appropriate platform and enabling the automatic flagging of potentially harmful content, this project has the potential to enhance online safety significantly. 

### _App Demo_ _([Test App](https://huggingface.co/spaces/sergey-hovhannisyan/toxic-tweets))_
https://user-images.githubusercontent.com/92773895/235402832-709ff881-bd2c-4ba6-ac41-fb96724a22a5.mp4



### _Setup Steps_ 
1. Clone Toxic Tweets repo
2. Create virtual environment (python -m venv venv)
3. Activate venv (venv\scripts\activate)
4. Install all required packages (pip install -r requirements.txt)
5. Run streamlit server (streamlit run app.py)
6. Input prompt & submit!

# _Training_
Data Preprocessing
In this project, the raw dataset of toxic tweets underwent tokenization and encoding using the batch_encode_plus() method from the Hugging Face tokenizer library. This method first tokenizes the text data into subword units and then encodes the subword units as numerical vectors that can be used as input to the neural network. Various parameters such as truncation, padding, and return_tensors were used to customize the encoding process. For example, truncation=True was used to ensure that tweets longer than the maximum sequence length were truncated, and padding=True was used to ensure that all sequences had the same length by padding with zeros. The encoded datasets were then converted to PyTorch tensors using the return_tensors='pt' parameter to ensure that the datasets were in the correct format to be used as inputs to the PyTorch model.

# _Model Building & Training:_
In this project, the pre-trained DistilBert model was fine-tuned for sequence classification using the DistilBertForSequenceClassification class from the Transformers library. This class was initialized with the pre-trained model name and the number of labels. The from_pretrained() method loads the pre-trained weights into the model, which are then fine-tuned on the labeled toxic tweets dataset. The fine-tuned model was moved to the specified device using the to() method. The training arguments for the model were specified using the TrainingArguments class from the Transformers library. A Trainer object was then created with the fine-tuned model, the training arguments, and the training and validation datasets. The train() method was called on the Trainer object to start the training process.
During training, the model was optimized using stochastic gradient descent with a warmup period and a learning rate of 5e-5. The training process was run for 2 epochs, with a batch size of 16 for training and 32 for validation. The training progress was logged to a directory specified by the logging_dir parameter, and the model was saved after each epoch using the save_strategy parameter. Only the most recent saved model was kept, with a total of 1 saved model allowed.

# _Deployment & Management:_
To efficiently deploy and manage the project, Docker containers were utilized. A Dockerfile was created that includes all the necessary dependencies and packages required for the project to run. This Dockerfile builds an image that can be run as a container on any Docker-supported platform. Hugging Face Spaces, a cloud-based platform that provides pre-configured Jupyter notebooks, data storage, and model hosting capabilities, was utilized to deploy the containerized project. This allowed for easy training and hosting of the model on the cloud.
To create a user interface for the model, Streamlit was utilized, a popular Python library for building web applications. A simple Streamlit app was created that allows users to enter a tweet and receive the predicted toxicity label from the model. This app was deployed to Hugging Face Spaces and made available to the public via a web URL.

# _Testing_
In the testing and analysis section, the test dataset was loaded and the comments and their corresponding labels were extracted. Then, the pre-trained DistilBERT model and tokenizer were loaded and the test set was encoded using the tokenizer. A dataset and DataLoader for the test set were created, and the batches were looped over to make predictions using the model. The predictions and labels were stored in lists, and the results were combined and analyzed to compute evaluation metrics.
Accuracy, precision, recall, and F1 score were used as evaluation metrics. The sklearn library was used to compute these metrics, and the results were output to the console. These metrics provide insight into the performance of the model on the test set and can be used to compare different models or variations of the same model.

# _Conclusion_
In conclusion, this project demonstrates the use of natural language processing and machine learning techniques to address the problem of online toxicity. The project involves data preprocessing, model building and training, deployment, and testing and analysis. The pre-trained DistilBERT model was fine-tuned for multi-label classification of toxic tweets, and the resulting model was deployed using Docker containers and Hugging Face Spaces. The Streamlit app provides a user-friendly interface for users to interact with the model and obtain toxicity predictions for their tweets. The evaluation metrics indicate that the model performs well on the test set, demonstrating its potential to improve online safety by automatically flagging potentially harmful content. Future work could involve incorporating additional data preprocessing & fine-tuning the model to improve its performance further.
