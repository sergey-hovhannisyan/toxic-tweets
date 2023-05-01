import streamlit as st
from src.evaluate import evaluate_prompt, model_list

st.title("Toxic Tweets")

# description of the project
st.info("The primary objective of this machine learning project in Natural Language Processing (NLP) is to accurately predict the toxicity level of tweets by utilizing fine-tuning techniques on a pre-trained language model. To achieve this, the project utilizes Docker containers for efficient deployment and management and Hugging Face Spaces and Transformers libraries for model building and training. The model is trained on a substantial dataset of labeled toxic tweets, allowing it to classify new tweets into five toxicity labels: \"severe toxic,\", \"obscene,\", \"threat,\", \"insult,\", and \"identity hate.\" By integrating the appropriate platform and enabling the automatic flagging of potentially harmful content, this project has the potential to enhance online safety significantly.")
# variables defined
sentiment_model_names = model_list()
section1, section2 = st.columns(2)

# function to predict the output
def predict(model_name, prompt):
    output = evaluate_prompt(model_name, prompt)
    with section2:
        st.table(output)
        st.success("Completed!")

# main code
with section1:
    st.header("Input")
    prompt = st.text_area("Prompt", "You fucking idiot. I will kill you!")
    model = st.selectbox("Select Model", sentiment_model_names)
    st.warning("albert & bert are self-supervised models, so possible relations are\
               LABLE_0|NEGATIVE, LABEL_1|POSITIVE.")
    st.button("Submit", on_click=lambda: predict(model, prompt))
with section2:
    st.header("Output")

    st.write("")
