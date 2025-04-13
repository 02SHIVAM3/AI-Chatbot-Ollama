import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM # allow us to interact with AI Model

#Load AI model from OLLama

llm = OllamaLLM(model="tinyllama")

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() #stores user-AI conversation history
    

# Define AI chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template="Previous conversation : {chat_history}\nUser : {question}\nAI : "
)


# Function to run AI chat with memory
def run_chain(question):
    # Retrieve the chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    
    # Run AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text,question=question))

    #store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response


# creating a web UI
st.title("🤖 AI Chatbot with Memory")
st.write("Ask Me Anything! ")
usrInp = st.text_input("📝 your question : ")
if usrInp:
    response = run_chain(usrInp)
    st.write(f"**You** {usrInp}")
    st.write(f"**AI** {response}")
    
# showing full chat history

st.subheader("🧾 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")

    

