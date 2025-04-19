import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# load AI Model
llm = OllamaLLM(model="tinyllama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate",160)

#speech recognition
recognizer = sr.Recognizer()

#function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()
    
def listen():
    with sr.Microphone() as source:
        st.write("\n🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"\n👂🏼 You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("😕 Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        st.write("⚠️ Speech Recognition Service Unavailable")
        return ""
    
prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template="Previous conversation : {chat_history}\nUser: {question}\nAI:"  
)

# function to process AI Responses

def run_chain(question):
    # retriev past chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    # Run AI Response generation 
    response = llm.invoke(prompt.format(chat_history=chat_history_text,question=question))

    #store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit Web UI
st.title("🤖 AI VOICE ASSISTANT")
st.write("\n🎙️ Click the button below to speak to me!")


# button to record voice input

if st.button("🎤 Listening"):
    usr = listen()
    if usr:
        response = run_chain(usr)
        st.write(f"**You** {usr}")
        st.write(f"**AI** {response}")
        speak(response)

# Display full chat history

st.subheader("🗒️ Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")
    


    