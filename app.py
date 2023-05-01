import streamlit as st
from src.evaluate import evaluate_prompt, model_list

st.title("Toxic Tweets")

# description of the project
st.info("This app utilizes a multi-head classifier that is fine-tuned to evaluate the toxicity level of a given prompt, providing 6 labels and their corresponding toxicity scores in descending order. By utilizing pre-trained language models, large labeled datasets, and fine-tuning techniques, the app helps determine if the prompt is toxic or not and contributes to enhancing online safety.")
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
