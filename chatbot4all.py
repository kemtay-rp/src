##### to run the ChatBot: streamlit run chatbot4all.py

# Import libraries
import streamlit as st
from openai import OpenAI
from datetime import datetime
from chat_model import ChatGPT4All, OpenAIGPT, GoogleGemini, Ollama
import pandas as pd
import io

# Initialize session state for `chat_model`
if "chat_model" not in st.session_state:
    st.session_state.chat_model = GoogleGemini(model="gemini-1.5-pro")  # Default model

def change_model():
    selected_model = st.session_state.radio_model 
    print(f"Selected model in on_change_handler: {selected_model}")
    match selected_model:
        case "GPT-4":
            st.session_state.chat_model = OpenAIGPT(model="gpt-4")
        case "Gemini":
            st.session_state.chat_model = GoogleGemini(model="gemini-2.0-flash-lite-preview-02-05")    #gemini-2.0-flash-lite-preview-02-05 ; gemini-1.5-pro
        case "Llama3.2":
            st.session_state.chat_model = Ollama(model="llama3.2")
            #st.session_state.chat_model = ChatGPT4All(model="Meta-Llama-38B-Instruct.Q4_0")
        case "Mistral":
            st.session_state.chat_model = Ollama(model="mistral")
        case "Gemma":
            st.session_state.chat_model = Ollama(model="gemma")
        case _:
            raise ValueError(f"Invalid model name: {selected_model}")

    st.session_state.chat_history = []  # Clear chat history
    print(f"st.session_state.chat_model: {st.session_state.chat_model}")
    # Initialize your chat model here
    #chat_model = ChatGPT4All(model="Meta-Llama-38B-Instruct.Q4_0")
    #chat_model = ChatGPT4All(model="Meta-Llama-3.1-8B-Instruct-128k-Q4_0")
    #chat_model = OpenAIGPT(model="gpt-4")
    #chat_model = GoogleGemini(model="gemini-1.5-pro-exp-0827")
    #chat_model = Ollama(model="llama3.2")
    #chat_model = Ollama(model="gemma")
    #chat_model = Ollama(model="mistral")


# Function to get response from the LLM
def chatbot_response(user_input):
    try:
        system_instruction = "You are a helpful AI assistant. The output shall be less than 200 words."
        #print(type(st.session_state.chat_history), st.session_state.chat_history[0]['message'])
        context_window = "|".join([dict['message'] for dict in st.session_state.chat_history])
        #print("context_window:", context_window)
        #prompt = { "system_message": system_instruction , "context_window": user_input }
        prompt = { "system_message": system_instruction , "context_window": context_window }
        response = st.session_state.chat_model.invoke(prompt, temperature=0.18)

        if "user\n" in response and "assistant\n" in response:
            responses = response.split("user\n")
            for i in range(len(responses)):
                print("chatbot_response():", responses[i].split("assistant")[1])

            return responses[i-1]
        else:
            return response
            
        # Call the new OpenAI API
        # completion = client.chat.completions.create(
        #     messages=[
        #         {"role": "system", "content": system_instruction},
        #         {"role": "user", "content": user_input}
        #     ],
        #     model="gpt-4o",
        # )
        # return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up page layout
st.set_page_config(
    page_title="ChatBot for All",
    layout="centered",
    initial_sidebar_state="auto",
)
#st.title('ChatBot for all')

# Sidebar
st.sidebar.title('ChatBot for All')
add_selectbox = st.sidebar.selectbox(
    "What would you like to test?",
    ("Prompt Injection", "Prompt Leaking", "Jailbreak")
)

# Initialize session state for `radio_model`
if "radio_model" not in st.session_state:
    st.session_state.radio_model = "Gemini"  # Default selection

if "session_title " not in st.session_state:
    st.session_state.session_title  = "ChatBot for All"  # Default selection

with st.sidebar:
    st.session_state.session_title = st.text_input("Enter a session title:", st.session_state.session_title)

    st.radio(
        "Choose a LLM model",
        ("Gemini", "GPT-4", "Llama3.2", "Mistral"),
        key="radio_model",  # Bind to session state key
        on_change=change_model  # This will be called on change
    )

    # File upload
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)  # Display as a DataFrame
            elif file_extension == "txt":
                # Handle text files
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                #st.text(string_data)
                st.session_state.chat_history.append({"role": "uploader", "message": string_data, "timestamp": datetime.now().strftime('%H:%M')})
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine='openpyxl') # Use openpyxl engine
                st.dataframe(df)
            elif file_extension == "json":
                import json
                try:
                    data = json.load(uploaded_file)
                    st.json(data) # Display as JSON
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
            else:
                st.write("Unsupported file format")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Clear chat to restart the session
    if st.button("Clear Chat"):
        #st.session_state.chat_history = []  # Clear chat history
        st.session_state.clear()
        st.rerun()

# Display chat histories
for chat in st.session_state.chat_history:        
    if chat["role"] != "uploader":
        align = "right" if chat["role"] == "user" else "left"
        bubble_color = "#d1e7dd" if chat["role"] == "user" else "#f1f1f1"

        #st.markdown(f"<p>{chat["message"]}</p>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="text-align: {align};">
                <div style="display: inline-block; background-color: {bubble_color}; padding: 10px; border-radius: 10px; margin: 5px; max-width: 80%; text-align: left;">
                    <p style="margin: 0;">{chat["message"]}</p>
                    <span style="font-size: 0.8em; color: gray;">{chat["timestamp"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# User input and chatbot response
if prompt := st.chat_input("Type your message here..."):
    user_message = prompt.strip()
    if user_message:  # Check if input is not empty
        chat_time = datetime.now().strftime('%H:%M')
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_message, "timestamp": chat_time})

        # Display user message immediately
        #with st.chat_message("user"):
        align = "right"
        bubble_color = "#d1e7dd"
        st.markdown(f"""
        <div style="text-align: {align};">
            <div style="display: inline-block; background-color: {bubble_color}; padding: 10px; border-radius: 10px; margin: 5px; max-width: 80%; text-align: left;">
                <p style="margin: 0;">{user_message}</p>
                <span style="font-size: 0.8em; color: gray;">{chat_time}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
        # Display a loading spinner while processing
        with st.spinner("Generating response..."):
            try:
                # Get bot response and add to chat history
                bot_response = chatbot_response(user_message)
                chat_time = datetime.now().strftime('%H:%M')
                st.session_state.chat_history.append({"role": "bot", "message": bot_response, "timestamp": chat_time})
            
                # Display bot response
                align = "left"
                bubble_color = "#f1f1f1"
                st.markdown(f"""
                <div style="text-align: {align};">
                    <div style="display: inline-block; background-color: {bubble_color}; padding: 10px; border-radius: 10px; margin: 5px; max-width: 80%; text-align: left;">
                        <p style="margin: 0;">{bot_response}</p>
                        <span style="font-size: 0.8em; color: gray;">{chat_time}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("An error occurred while processing your request.")
                st.error(str(e))

