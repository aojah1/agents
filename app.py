import streamlit as st
import os

st.title("🤖 ReAct - MultiAgent Chatbot")

# Display the Multi-Agent Chatbot architecture image
st.image("multi-agent-react.png", caption="Multi-Agent Chatbot Architecture")

# Run the multi-agent chatbot script
script_path = "multi-agent-chatbot.py"
if os.path.exists(script_path):
    exec(open(script_path).read())

#st.write("session_id :" + session_id)
# Call functions from the script
if "test_react" in globals():
    user_query = st.text_input("Ask a question:", "What are the tax implications for corporations in FY 2025?")
    if st.button("Submit Query"):
        response = test_react(user_query)
        st.subheader("Response:")
        st.write(response)
else:
    st.error("Error: Chatbot script could not be loaded.")
