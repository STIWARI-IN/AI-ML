import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

groq_api_key = st.secrets['GROQ_SECRET_KEY']
os.environ['GROQ_API_KEY'] = groq_api_key

from langchain_groq import ChatGroq
llm = ChatGroq(model='llama3-8b-8192', temperature=0)


def hospital_advisor(place):
    # Generate prompt for hospital recommendation
    prompt_template_name = PromptTemplate(
        input_variables=['place'],
        template='I want to go {place} for medical treatment'
    )

    # Chain for generating the place name
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='place_name')

    # Generate hospital suggestions
    prompt_template_items = PromptTemplate(
        input_variables=['place_name'],  # This must match the output_key from name_chain
        template='Suggest some famous hospitals in {place_name}. Provide private and government hospitals separately.'
    )

    # Chain for generating best hospitals
    items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='best_hospital')

    # Sequential chain to run both parts
    seq_chain = SequentialChain(
        chains=[name_chain, items_chain],
        input_variables=['place'],
        output_variables=['place_name', 'best_hospital']
    )

    # Get result by passing the place
    result = seq_chain({'place': place})
    return result


# Streamlit UI code
st.set_page_config(page_title="Hospital Advisor", page_icon="🏥", layout="wide")

st.title('AI Based Hospital Advisor 🏥')

# Sidebar input for place
input_user_text = st.sidebar.text_input('Enter the Place for Treatment! ✈️')

# Handle search button press
if st.sidebar.button('Search'):
    if input_user_text:
        # Get the hospital recommendations based on the place entered
        response = hospital_advisor(input_user_text)
        
        # Display the result: Place name and hospital list
        st.header(f"Best Hospitals in {response['place_name']}")
        list_item = response['best_hospital'].split(',')
        st.write('**List of Hospitals (Private & Government)**')
        for item in list_item:
            st.write(item)
    else:
        st.warning('Please enter a valid place for treatment.')
