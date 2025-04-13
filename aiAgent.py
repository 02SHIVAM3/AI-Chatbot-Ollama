from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM # allow us to interact with AI Model

#Load AI model from OLLama

llm = OllamaLLM(model="tinyllama")

# Initialize Memory
chat_history = ChatMessageHistory() #stores users-AI convo history

# Define AI hat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template="Previous conversation : {chat_history}\nUser : {question}\nAI : "
)


# Function to run AI chat with memory
def run_chain(question):
    # Retrieve the chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
    
    # Run AI response generation
    response = res = llm.invoke(prompt.format(chat_history=chat_history_text,question=question))

    #store new user input and AI response in memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response
    
# Interactive CLI Chatbot
print("\n🤖 AI Chatbot with Memory")
print("\nType 'exit' to stop")
while True:
    usrInp = input("\n\n📝 your question : ")
    if(usrInp.lower() == "exit"):
        print("👋 GOOD BYE :) ")
        break
    res = run_chain(usrInp)
    print("\n🤖 AI Response : ", res)

    
