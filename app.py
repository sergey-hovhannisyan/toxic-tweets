import streamlit as st
from src.evaluate import evaluate_prompt, model_list

st.title("Toxic Tweets")

st.info("This NLP machine learning project aims to predict the toxicity level of input tweets using fine-tuning techniques on a pre-trained language model. The project utilizes Docker containers for efficient deployment and management, while Hugging Face Spaces and Transformers provide the necessary libraries and tools for building and training the model. The model is trained on a large dataset of labeled toxic tweets to enable it to classify new input tweets as toxic or non-toxic. This project can help improve online safety by automatically flagging potentially harmful content.")

# variables defined
sentiment_model_names = model_list()
section1, section2 = st.columns(2)

# functions
def predict(model_name, prompt):
    output = evaluate_prompt(model_name, prompt)[0]
    label = output["label"]
    score = output["score"]
    with section2:
        st.write("Label:", label)
        st.write("Score:", score)
        st.success("Completed!")


with section1:
    st.header("Input")
    prompt = st.text_area("Prompt", placeholder="Eat Up Every Moment")
    model = st.selectbox("Select Model", sentiment_model_names)
    st.warning("albert & bert are self-supervised models, so possible relations are\
               LABLE_0|NEGATIVE, LABEL_1|POSITIVE.")
    st.button("Submit", on_click=lambda: predict(model, prompt))
with section2:
    st.header("Output")

    st.write("")
