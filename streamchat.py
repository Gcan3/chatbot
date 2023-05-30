# Importing library
import streamlit as st
import pandas as pd
import time

from data import init_vector_db, embed_vectors, Model, API_KEY, ENVIRONMENT
from streamlit_chat import message
from io import StringIO

# Initialize vector database
index = init_vector_db(API_KEY,
                           ENVIRONMENT,
                           'course-test-index',
                           768,
                           metric = 'dotproduct')

# Connecting with CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main Content Header
st.header("FTSC ChatBot")

#trying to reformat the input text as a query to feed into the model
def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text 

#implementing spinner to give the user indication of loading contents
with st.spinner('Loading, Please wait a moment...'):
    time.sleep(1)
    
    # Main Content Layout 
    sideUtil, main = st.columns([5,7])

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
                "logo.png",
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

    #Sidebar Util
    with sideUtil:
        # File uploader
        uploaded_file = st.file_uploader("Upload a csv file (IMPORTANT):")
        
        #Actions when the file is submitted
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # To read file as string:
            string_data = stringio.read()
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            embedder = embed_vectors(dataframe, 
                                    'context',
                                    'sentence-transformers/msmarco-distilbert-base-tas-b',
                                    index,
                                    batch_size = 5)

        
        # Select_sliders for answer length and num_beam
        min_ans = st.select_slider('Minimum length of the answer: ',
                        options=[10, 15, 20])
        max_ans = st.select_slider('Maximum length of the answer: ', 
                        options=[50, 80, 100])
        numBeam = st.select_slider('Beam search (probable word search): ',
                        options=[1, 2, 3, 4, 5, 6, 7, 8])
        
        # Sliders for top_k and num_beams
        topk = st.slider('Number of context:', 0, 10, 5)

    # Session state to store the previous chat messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    #Main Content
    with main:
        user_input = get_text()
        
        # ignoring streamlit error using try-except syntax
        try:
            model = Model('google/flan-t5-large',
                        embedder,
                        index)
            query = model.make_query(user_input, 'context', topk) # fetches context from vector db and reformats it
            answer = model.generate_answer(query) # generate answer
            
            #appending the chat history into their specific dictionaries
            if user_input:
                st.session_state.past.append(user_input)
                st.session_state.generated.append(answer)
        except:
            #error message before beginning
            st.error("Please enter the CSV file to activate our chat bot")
            pass
            
            
        #displaying the generated output by the model IF there is an output
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
