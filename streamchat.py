# Importing library
import streamlit as st
import data

from streamlit_chat import message
from io import StringIO

# Connecting with CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main Content Header
st.header("FTSC ChatBot")

# Main Content Layout 
sideUtil, main = st.columns([3,6])

sideUtil.success("Utilities")
main.success("Chatbox")

#sidebar container
with st.sidebar:
    #sidebar header
    st.subheader("Hello! ¡Hola! Guten Tag! Bonjour!")
    #setting layout of contents
    col1, col2, col3 = st.columns([7,10,7])

    #making blank statement
    with col1:
        st.write("")

    #Putting the image at the middle row
    with col2:
        st.image(
            "resource/logo.png",
            use_column_width="auto",
        )

    #making blank statement
    with col3:
        st.write("")
    
    #creating a new line for text
    "---"
    st.write("""**Welcome to our FTSC chatbot web application.  
             Make sure to please fill out some requirements before starting**""")
    st.write("""We are striving to make our teachings and discoveries clearer in parallel with our technologies and ideas of today. With the help of our chatbot, we can easily clarify and strike out the remaining factors of doubts at an online environment.""")
    
#Sidebar
with sideUtil:
    uploaded_file = st.file_uploader("Upload a csv file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

#Main COntent
with main:
    user_input = get_text()
    
    if user_input:
        #trying to attach it to the main class
        output = data.main
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        
#displaying the generated output by the model IF there is an output
if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')