import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
import os

## Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')
# Funciton to get respone from LLAma 2 model

def getLLamaresponse(input_text):
    # calling llama 3 -8b model with groq
    llm = ChatGroq(temperature=0.01,model="llama3-8b-8192",groq_api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_template(
    """
        Please analyze the {input_text} and extract the following information:
            * Person's name
            * Email address
            * Job role
            * Availability 
            * Reason they are intersted in Keelworks
        In case no information is found, please indicate that.
    """)

    # generate the response from llama 2 model
    chain = prompt | llm
    print(chain)
    response = chain.invoke({"input_text":input_text})
    print(response.content)
    return response


st.set_page_config(page_title='Keelworks Support Chatbot',
                   layout = 'centered',
                   initial_sidebar_state='collapsed')

st.header("Keelworks Support Chatbot")

input_text = st.text_input("Enter your query here ")

submit = st.button('Submit')

# Final Response

if submit:
    response = getLLamaresponse(input_text)
    st.write(response.content)



