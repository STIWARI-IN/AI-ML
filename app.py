
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

#from secret_key import groq_api_key

groq_api_key=st.secrets['GROQ_API_KEY']
os.environ['GROQ_API_KEY']=groq_api_key

from langchain_groq import ChatGroq
llm=ChatGroq(model='llama3-8b-8192',temperature=0)



def travel_advisor(State):
   
    prompt_template_name = PromptTemplate(
    input_variables=['State'],
    template= 'I want to travel {State}'
    )
   


    name_chain=LLMChain(llm=llm, prompt=prompt_template_name, output_key='state_name')

    prompt_template_items = PromptTemplate(
    input_variables=['state_name'],
    template= 'Suggest some famous sight seeing for {state_name}. Return it as a comma separated'
    )

    items_chain=LLMChain(llm=llm,prompt=prompt_template_items, output_key='sight_seeing_items')

    seq_chain = SequentialChain(
    chains=[name_chain, items_chain],
    input_variables=['State'],
    output_variables=['state_name','sight_seeing_items']
    )

    result=seq_chain({'State':State})
    return result

if __name__ == '__main__':
    print(travel_advisor('Punjab'))


import streamlit as st
st.title('Travel Advisor 🛣️')
input_user_text=st.sidebar.text_input('Enter State Name Where You Want To Visit!')
st.sidebar.button('search')


if input_user_text:
    response = travel_advisor(input_user_text)
    st.header(response['state_name'])
    list_item=response['sight_seeing_items'].split(',')
    st.write('**Famous Sight Seeing Places**')
    for item in list_item:
        st.write(item)
