#Added to fix error on streamlit
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
########


import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import re
import streamlit as st

# UI Setup 
st.set_page_config(
    page_title="Amica AI Assistant",
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS 
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #191970; /* Midnight Blue for general text visibility */
        font-family: 'Roboto', sans-serif;
    }

    /* Ensure all markdown text is dark, including in headers/intros */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] em,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #191970 !important;
    }

    /* Header Styling */
    .st-emotion-cache-nahz7x { /* Streamlit title element */
        color: #1a73e8; /* Google Blue */
        font-weight: 600;
        text-align: center;
    }
    /* This specific class might be for a subheader/markdown, ensure its text is also dark */
    .st-emotion-cache-18jva2u {
        text-align: center;
        color: #191970 !important; /* Midnight Blue */
    }

    /* Chat Message Styling */
    .st-emotion-cache-1r6dm1c { /* Wrapper for chat messages */
        border-radius: 15px;
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        line-height: 1.5;
        font-size: 16px;
        /* Default color for the chat message div itself */
        color: #191970 !important;
    }
    /* Targeting the inner paragraph and other text elements within chat messages for robust text color */
    .st-emotion-cache-1r6dm1c p,
    .st-emotion-cache-1r6dm1c strong,
    .st-emotion-cache-1r6dm1c em,
    .st-emotion-cache-1r6dm1c div[data-testid="stMarkdownContainer"] {
        color: #191970 !important; /* Midnight Blue for text inside messages */
    }

    .st-emotion-cache-1r6dm1c.st-emotion-cache-10q71qg { /* User message background */
        background-color: #e3f2fd; /* Light Blue 100 */
        border-bottom-right-radius: 3px;
        margin-left: 20%; /* Push user messages to the right */
    }
    .st-emotion-cache-1r6dm1c:not(.st-emotion-cache-10q71qg) { /* Bot message background */
        background-color: #ffffff; /* White */
        border-bottom-left-radius: 3px;
        margin-right: 20%; /* Push bot messages to the left */
    }

    /* Avatar Styling */
    .st-emotion-cache-1r6dm1c.st-emotion-cache-10q71qg > div > div:first-child { /* User avatar */
        background-color: #4285f4; /* Google Blue */
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 12px;
        font-size: 18px;
    }
    .st-emotion-cache-1r6dm1c:not(.st-emotion-cache-10q71qg) > div > div:first-child { /* Bot avatar */
        background-color: #6c757d; /* Grey */
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 12px;
        font-size: 18px;
    }

    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 20px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 16px;
        background-color: white;
        color: #191970; /* Midnight Blue for input text */
    }
    .stTextInput > label {
        display: none; /* Hide the default label for chat input */
    }

    /* Send Button (if using a separate button) */
    .stButton > button {
        border-radius: 25px;
        background-color: #4285f4; /* Google Blue */
        color: white;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        font-size: 16px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #357ae8; /* Darker blue on hover */
    }

    /* Streamlit container padding adjustment */
    .st-emotion-cache-1c7y2vl { /* Main container padding */
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Setup
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Error: GEMINI_API_KEY not found in environment variables or .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_INSTRUCTION_AMICA = os.getenv("SYSTEM_INSTRUCTION_AMICA")
TELECOM_BILLING_KEYWORDS = [k.strip() for k in os.getenv("TELECOM_BILLING_KEYWORDS", "").split(',') if k.strip()]

# Load Synthetic Data from CSV
SYNTHETIC_DATA_PATH = os.getenv("SYNTHETIC_DATA_PATH", "synthetic_telco_data.csv")
synthetic_data_df = pd.DataFrame() # Initialize as empty DataFrame

try:
     
    synthetic_data_df = pd.read_csv(SYNTHETIC_DATA_PATH, dtype={'Subscriber_ID': str})
    
    synthetic_data_df.columns = synthetic_data_df.columns.str.strip()

    
    def convert_subscriber_id_to_full_string(s_id):
        if pd.isna(s_id): # Handle NaN values
            return None
        try:

            return str(int(float(s_id)))
        except (ValueError, TypeError):

            return str(s_id)

    synthetic_data_df['Subscriber_ID'] = synthetic_data_df['Subscriber_ID'].apply(convert_subscriber_id_to_full_string)
    


except FileNotFoundError:
    st.warning(f"Warning: `{SYNTHETIC_DATA_PATH}` not found. Amica's personalized features will be limited.")
except Exception as e:
    st.warning(f"Error loading or formatting synthetic data from `{SYNTHETIC_DATA_PATH}`: {e}. Amica's personalized features will be limited.")


#Session State Initialization 
if "chat_session" not in st.session_state:
    try:
        if SYSTEM_INSTRUCTION_AMICA:
            model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION_AMICA)
        else:
            st.warning("SYSTEM_INSTRUCTION_AMICA is None. Running model without specific system instruction.")
            model = genai.GenerativeModel(MODEL_NAME)
        
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.amica_subscriber_id = None
        st.session_state.amica_subscriber_data = None
        st.session_state.initial_amica_query = None
        st.session_state.messages = [] # To store messages for display in UI
         
        st.session_state.messages.append({"role": "Amica", "content": "I am Amica (v2), your dedicated customer care assistant for telecom subscribers. To provide you with a personalized telecom experience, may I have your Subscriber ID? It’s a 13-digit number starting with 44."})
    except Exception as e:
        st.error(f"Failed to initialize Amica model: {e}")
        st.stop()

#Helper to get subscriber data
def get_subscriber_data(subscriber_id):
    """Retrieves subscriber data from the synthetic DataFrame."""
    if synthetic_data_df.empty:
        return False 
    if 'Subscriber_ID' not in synthetic_data_df.columns:
         
        st.error("Internal Error: 'Subscriber_ID' column not found after data loading.")
        return False
    
    # Ensure the subscriber_id being searched is also a string (already handled by apply func)
    data = synthetic_data_df[synthetic_data_df['Subscriber_ID'] == subscriber_id]
    if not data.empty:
        st.session_state.amica_subscriber_data = data.iloc[0].to_dict()
        return True
    return False

# Bot Information (Top of UI) 
st.title("Amica AI Assistant 🤖")
st.markdown("""
    Welcome to Amica (v2), your personalized telecom customer care assistant!
    I'm here to help you with your account, billing, plans, and services.
    
    **To get started, please provide your Subscriber ID (a 13-digit number starting with 44).**
    
    **Example Prompts:**
    * "My account number is 4412345678901."
    * "What's my data usage?"
    * "I have a question about my last bill."
    * "Can you help me with roaming?"
    * "What's my current plan?"
""")
st.markdown("---")

#Display Chat History
for message in st.session_state.messages:
    
    if message["role"] == "You":
        with st.chat_message("You", avatar="👤"):  
            st.markdown(message["content"])
    else: # Amica
        with st.chat_message("Amica", avatar="🤖"):  
            st.markdown(message["content"])

 
user_input = st.chat_input("Ask Amica...")

if user_input:
     
    st.session_state.messages.append({"role": "You", "content": user_input})
     
    with st.chat_message("You", avatar="👤"):
        st.markdown(user_input)

    response_content = "" # Accumulate Amica's response here

    # Subscriber ID handling logic
    if st.session_state.amica_subscriber_id is None:
         
        subscriber_id_match = re.search(r'44\d{11}', user_input)
        if subscriber_id_match:
            extracted_id = subscriber_id_match.group(0)
            
             
            if get_subscriber_data(extracted_id): 
                st.session_state.amica_subscriber_id = extracted_id  
                response_content += f"Thank you, Subscriber ID **{st.session_state.amica_subscriber_id}** confirmed. How can I assist you with your telecom needs today?\n\n"
                
                if st.session_state.amica_subscriber_data:
                    monthly_data = st.session_state.amica_subscriber_data.get('Monthly Data Usage (GB)', 'N/A')
                    voice_minutes = st.session_state.amica_subscriber_data.get('Voice Minutes', 'N/A')
                    last_recharge_amount = st.session_state.amica_subscriber_data.get('Last Recharge ($)', 'N/A')
                    ott_usage = st.session_state.amica_subscriber_data.get('OTT Usage', 'N/A')
                    roaming_status_display = 'Active' if st.session_state.amica_subscriber_data.get('Roaming Used') == 'Yes' else 'Inactive'
                    device = st.session_state.amica_subscriber_data.get('Device', 'N/A')
                    tenure = st.session_state.amica_subscriber_data.get('Tenure (months)', 'N/A')

                    #response_content += f"Your Monthly Data Usage is **{monthly_data} GB** and Voice Minutes are **{voice_minutes}**. Your last recharge amount was **${last_recharge_amount}**.\n"
                    #response_content += f"Your primary OTT usage includes **{ott_usage}**. Your roaming status is **{roaming_status_display}**.\n"
                    #response_content += f"You've been with us for **{tenure} months** and use a **{device}**.\n\n"
                
                
                if st.session_state.amica_subscriber_data:
                    context_message = (
                        f"Customer details provided: Subscriber ID {st.session_state.amica_subscriber_id}, "
                        f"Monthly Data Usage: {st.session_state.amica_subscriber_data.get('Monthly Data Usage (GB)')} GB, "
                        f"Voice Minutes: {st.session_state.amica_subscriber_data.get('Voice Minutes')}, "
                        f"ARPU: {st.session_state.amica_subscriber_data.get('ARPU ($)')}, "
                        f"Roaming Status: {st.session_state.amica_subscriber_data.get('Roaming Used')}, "
                        f"International Calls Per Week: {st.session_state.amica_subscriber_data.get('Intl Calls/Wk')}, "
                        f"OTT Apps: {st.session_state.amica_subscriber_data.get('OTT Usage')}, "
                        f"Churn Score: {st.session_state.amica_subscriber_data.get('Churn Score')}, "
                        f"Complaints Count: {st.session_state.amica_subscriber_data.get('Complaints Count')}, "
                        f"Last Recharge Amount: {st.session_state.amica_subscriber_data.get('Last Recharge ($)')}$, "
                        f"Top-up Frequency: {st.session_state.amica_subscriber_data.get('Top-up Freq')}, "
                        f"Device: {st.session_state.amica_subscriber_data.get('Device')}, "
                        f"Tenure: {st.session_state.amica_subscriber_data.get('Tenure (months)')} months. "
                        "Remember your role as Amica. Do not reveal churn score, ARPU or complaints count directly to the customer. "
                        "Only use this data for internal decision-making to tailor your responses. "
                        "If asked for 'current balance' or similar, respond by stating that you can only provide 'Last Recharge Amount' as per available data."
                    )
                     
                    st.session_state.chat_session.history.append({
                        "parts": [{"text": context_message}],
                        "role": 'model'
                    })
                
                # Now, send the initial_amica_query to the LLM with the newly established context
                if st.session_state.initial_amica_query:
                    try:
                        llm_response = st.session_state.chat_session.send_message(st.session_state.initial_amica_query, stream=True)
                        for chunk in llm_response:
                            response_content += chunk.text
                        st.session_state.initial_amica_query = None # Clear it after sending
                    except Exception as e:
                        response_content += f"An error occurred while getting a response for your initial query: {e}"
                        response_content += "Please try again or check your API key/network connection."
                
            else:  
                response_content = f"Subscriber ID **{extracted_id}** not found in our synthetic system. I can provide general information, but personalized services will be limited. Please try one of the example IDs like **4412345678901**." # Updated example ID
                # Reset amica_subscriber_id to None if not found, so it keeps asking
                st.session_state.amica_subscriber_id = None 
        elif user_input.lower() in ["no", "i don't want to give it"]:
            response_content = "Understood. My services will be limited without your Subscriber ID, but I can provide general telecom information. How can I help?"
        else:
             
            personalized_keywords = ["balance", "bill", "plan", "my account", "my usage", "my details", "my data", "my current", "my last", "my phone", "my services"]
            is_personalized_query = any(keyword in user_input.lower() for keyword in personalized_keywords)
            if is_personalized_query:
                response_content = "To provide you with specific information like balance or plan details, I need your Subscriber ID. It’s a 13-digit number starting with 44. Can you please provide it?" # Reverted prompt
            else:
                 
                try:
                    llm_response = st.session_state.chat_session.send_message(user_input, stream=True)
                    for chunk in llm_response:
                        response_content += chunk.text
                except Exception as e:
                    response_content += f"An error occurred while getting a response: {e}"
                    response_content += "Please try again or check your API key/network connection."
    else: 
        try:
            llm_response = st.session_state.chat_session.send_message(user_input, stream=True)
            for chunk in llm_response:
                response_content += chunk.text
        except Exception as e:
            response_content += f"An error occurred while getting a response: {e}"
            response_content += "Please try again or check your API key/network connection."
            
     
    with st.chat_message("Amica", avatar="🤖"):
        st.markdown(response_content)
     
    st.session_state.messages.append({"role": "Amica", "content": response_content})
