import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

groq_api_key = st.secrets['GROQ_SECRET_KEY']
os.environ['GROQ_API_KEY'] = groq_api_key

from langchain_groq import ChatGroq
llm = ChatGroq(model='llama3-8b-8192', temperature=0)


def travel_advisor(State):
    prompt_template_name = PromptTemplate(
        input_variables=['State'],
        template='I want to travel {State}'
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='state_name')

    prompt_template_items = PromptTemplate(
        input_variables=['state_name'],
        template='Suggest some famous sightseeing for {state_name}. Return it as a comma separated'
    )

    items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='sight_seeing_items')

    seq_chain = SequentialChain(
        chains=[name_chain, items_chain],
        input_variables=['State'],
        output_variables=['state_name', 'sight_seeing_items']
    )

    result = seq_chain({'State': State})
    return result


# Streamlit UI with attractive layout and styles
st.set_page_config(page_title="Travel Advisor", page_icon="🌍", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
        }
        .sidebar .sidebar-content {
            background-color: #f0f8ff;
            padding: 20px;
        }
        .title {
            color: #2e8b57;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-top: 50px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .input-box {
            background-color: #ffebcd;
            border: 2px solid #ff6347;
            padding: 12px;
            font-size: 16px;
            border-radius: 5px;
        }
        .input-box:focus {
            border: 2px solid #8a2be2;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 18px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .result-card h4 {
            font-size: 24px;
            color: #2e8b57;
        }
        .result-card ul {
            list-style-type: none;
            padding-left: 0;
        }
        .result-card ul li {
            margin: 10px 0;
            font-size: 16px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>Travel Advisor 🌍</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Find famous sightseeing places based on your travel destination</p>", unsafe_allow_html=True)

# Sidebar with input and buttons
input_user_text = st.sidebar.text_input('Enter Place Where You Want To Visit! ✈️', key='place_input', placeholder='E.g., Punjab, Goa, Paris')
search_button = st.sidebar.button('Search', key='search_button', help="Click to find famous places to visit")
clear_button = st.sidebar.button('Clear Search', key='clear_button', help="Clear the search and start over")

# Functionality for Search Button
if search_button and input_user_text:
    response = travel_advisor(input_user_text)
    st.markdown(f"<div class='result-card'><h4>{response['state_name']}</h4><ul>", unsafe_allow_html=True)
    list_item = response['sight_seeing_items'].split(',')
    for item in list_item:
        st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
    st.markdown("</ul></div>", unsafe_allow_html=True)

# Functionality for Clear Button
if clear_button:
    st.session_state['place_input'] = ""  # Reset the input field in session state
    st.experimental_rerun()  # Rerun the app to clear the results

