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
            background: linear-gradient(135deg, #7F7FD5, #86A8E7, #91EAE4);
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        .title {
            color: #fff;
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            margin-top: 50px;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.4);
        }
        .description {
            text-align: center;
            font-size: 20px;
            color: #fff;
            margin-top: 10px;
        }
        .input-box {
            background-color: #ffffff;
            border: 2px solid #ff6347;
            padding: 12px;
            font-size: 16px;
            border-radius: 10px;
            width: 100%;
        }
        .input-box:focus {
            border: 2px solid #8a2be2;
            outline: none;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px 25px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #45a049;
        }
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
            animation: fadeIn 0.5s ease-out;
        }
        .result-card h4 {
            font-size: 26px;
            color: #2e8b57;
            text-align: center;
            font-weight: bold;
        }
        .result-card ul {
            list-style-type: none;
            padding-left: 0;
        }
        .result-card ul li {
            margin: 10px 0;
            font-size: 18px;
            color: #555;
            text-align: center;
        }
        @keyframes fadeIn {
            0% {
                opacity: 0;
            }
            100% {
                opacity: 1;
            }
        }
        .search-section {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 50px;
        }
        .search-section input {
            width: 60%;
            padding: 12px;
            font-size: 16px;
            border-radius: 12px;
            border: 2px solid #ff6347;
            outline: none;
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>Travel Advisor 🌍</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Find famous sightseeing places based on your travel destination</p>", unsafe_allow_html=True)

# Sidebar with input and buttons
input_user_text = st.sidebar.text_input('Enter Place Where You Want To Visit! ✈️', key='place_input', placeholder='E.g., Punjab, Goa, Paris')
search_button = st.sidebar.button('Search', key='search_button', help="Click to find famous places to visit")

# Functionality for Search Button
if search_button and input_user_text:
    response = travel_advisor(input_user_text)
    st.markdown(f"<div class='result-card'><h4>{response['state_name']}</h4><ul>", unsafe_allow_html=True)
    list_item = response['sight_seeing_items'].split(',')
    for item in list_item:
        st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
    st.markdown("</ul></div>", unsafe_allow_html=True)
