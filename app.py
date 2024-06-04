import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

import os

import json
import re

# Load the predefined questions and answers from the JSON file
# with open('predefined_qa.json', 'r') as f:
#     predefined_qa_data = json.load(f)

# Create a dictionary from the loaded data
# predefined_qa = {qa['pattern']: qa['answer'] for qa in predefined_qa_data['questions']}

def get_predefined_response(input_text):
    """
    Check if the input text matches any predefined question.
    If a match is found, return the corresponding predefined answer.
    Otherwise, return None.
    """
    for pattern, answer in predefined_qa.items():
        if re.search(pattern, input_text, re.IGNORECASE):
            return answer
    return None

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
            * Company's name they are associate with
            * Person's identity
            * Email address
            * Job role
            * Availability 
            * Reason they are intersted in meeting
        Once you identify this please answer this in a table mentioning "Yes" or "No" for above extarcted information. 
        In case no information is found, please indicate that.
    """)

    # generate the response from llama 2 model
    chain = prompt | llm
    # print(chain)
    response = chain.invoke({"input_text":input_text})
    # print(response.content)

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



