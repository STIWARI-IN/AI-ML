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

# Custom CSS for light theme background image
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f0f5;  /* Light gray background for the light theme */
            color: #333;  /* Dark text for contrast */
        }
        .title {
            color: #333;
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            margin-top: 50px;
            text-shadow: 2px 2px 10px rgba(255, 255, 255, 0.6);
        }
        .description {
            text-align: center;
            font-size: 20px;
            color: #555;
            margin-top: 10px;
        }
        .input-box {
            background-color: rgba(255, 255, 255, 0.8);
            border: 2px solid #007bff;
            padding: 12px;
            font-size: 16px;
            border-radius: 12px;
            width: 100%;
            color: #333;
        }
        .input-box:focus {
            border: 2px solid #0056b3;
            outline: none;
        }
        .button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 12px 25px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #0056b3;
        }
        .result-card {
            background-color: rgba(0, 123, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            margin-top: 30px;
        }
        .result-card h4 {
            font-size: 26px;
            color: #FFF;
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
            color: #FFF;
            text-align: center;
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
            border: 2px solid #007bff;
            outline: none;
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>AI Based Hospital Advisor 🏥</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Find the best hospitals for treatment in your desired place</p>", unsafe_allow_html=True)

# Initialize session state for input_user_text if not already present
if "input_user_text" not in st.session_state:
    st.session_state.input_user_text = ""

# Sidebar input for place
input_user_text = st.sidebar.text_input('Enter the Place for Treatment! ✈️', value=st.session_state.input_user_text)

# Button to clear the input text
if st.sidebar.button('Clear Search'):
    st.session_state.input_user_text = ""  # Reset the input text

# Handle search button press
if st.sidebar.button('Search'):
    if input_user_text:
        # Get the hospital recommendations based on the place entered
        response = hospital_advisor(input_user_text)
        
        # Display the result: Place name and hospital list
        if response["place_name"] == "Invalid Place":
            st.error(response["best_hospital"])
        else:
            st.header(f"Best Hospitals in {response['place_name']}")  # Primary Header for the results
            list_item = response['best_hospital'].split(',')
            st.write('**List of Hospitals (Private & Government)**')  # Optional subheader
            for item in list_item:
                st.write(item)
    else:
        st.warning('Please enter a valid place for treatment.')  # Warning for empty input
