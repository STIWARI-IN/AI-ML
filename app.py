import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

groq_api_key = st.secrets['GROQ_SECRET_KEY']
os.environ['GROQ_API_KEY']=groq_api_key

from langchain_groq import ChatGroq
llm=ChatGroq(model='llama3-8b-8192',temperature=0)



def hospital_advisor(Place):
   
    prompt_template_name = PromptTemplate(
    input_variables=['place'],
    template= 'I want to go {State} for medical treatment'
    )
   


    name_chain=LLMChain(llm=llm, prompt=prompt_template_name, output_key='place_name')

    prompt_template_items = PromptTemplate(
    input_variables=['place_name'],
    template= 'Suggest some famous hospital in {state_name}. Provide all private and Government hospital separately'
    )

    items_chain=LLMChain(llm=llm,prompt=prompt_template_items, output_key='hospitals')

    seq_chain = SequentialChain(
    chains=[name_chain, items_chain],
    input_variables=['place'],
    output_variables=['place_name','hospitals']
    )

    result=seq_chain({'State':State})
    return result

if __name__ == '__main__':
    print(hospital_advisor('Punjab'))


import streamlit as st
st.title('AI Based Hospital Advisor 🏥')
input_user_text=st.sidebar.text_input('Enter the Place for Treatment!! ✈️')
st.sidebar.button('search')


if input_user_text:
    response = travel_advisor(input_user_text)
    st.header(response['place_name'])
    list_item=response['sight_seeing_items'].split(',')
    st.write('**List of All Hospitals**')
    for item in list_item:
        st.write(item)
