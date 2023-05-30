#importing libraries
import pinecone
import torch
import pandas as pd
import streamlit as st

from io import StringIO
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm

API_KEY ='7ccbdb2f-63a9-4a62-b739-5d0f117cc3a8'
ENVIRONMENT = 'asia-northeast1-gcp'

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
        output = query
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        
#displaying the generated output by the model IF there is an output
if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    






# --------------------SEAN'S CODE------------------------
class VectorDB:
    def __init__(self, api_key, environment):
        self.api_key = api_key
        self.environ = environment

        pinecone.init(
            api_key = self.api_key,
            environment = self.environ
        )

    def create_index(self, index_name, dims, metric = 'cosine'):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name = index_name, 
                dimension = dims,
                metric = metric
            )
        index = pinecone.Index(index_name)
        print(index)
        return index

class VectorEmbedding:
    def __init__(self, model_ref, index):
        # agnostic code -- run tensors on gpu if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.index = index
        self.embedder = SentenceTransformer(model_ref, device)
    
    def create_embeddings(self, dataframe, context_column_name, batch_size = 1):
        for idx in tqdm(range(0, len(dataframe))):
            end_idx = min(idx + batch_size, len(dataframe))
            batch = dataframe.iloc[idx:end_idx]
            
            vector_embeddings = self.embedder.encode(batch[context_column_name].tolist()).tolist()
            metadata = batch.to_dict(orient = 'records')
            unique_ids = [f'{i}' for i in range(idx, end_idx)]

            vectors = list(zip(unique_ids, 
                            vector_embeddings, 
                            metadata))

            _ = self.index.upsert(vectors = vectors)
        print(self.index.describe_index_stats())

class Model:
    def __init__(self, 
                 model_checkpoint, 
                 embedder,
                 index):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(self.device)

        self.embedder = embedder
        self.index = index
    
    def query_db(self, query, top_k):
        encoded_query = self.embedder.encode([query]).tolist()
        encoded_context = self.index.query(encoded_query, top_k = top_k, include_metadata = True)
        print(encoded_context)
        return encoded_context
    
    def format_query(self, query, context_column_name, context):
        context = [f"<P> {meta['metadata'][context_column_name]}" for meta in context]
        context = ' '.join(context)
        query = f'question: {query} context: {context}'
        return query   

    def make_query(self, query, context_column_name, top_k = 3):
        context = self.query_db(query, top_k = top_k)
        query = self.format_query(query, context_column_name, context['matches'])
        return query     
    
    def generate_answer(self, query, min_length = 20, max_length = 50):
        inputs = self.tokenizer([query], max_length = 1024, return_tensors = 'pt', truncation = True).to(self.device)
        ids = self.generator.generate(inputs['input_ids'], 
                                      num_beams = 2, 
                                      min_length = min_length, 
                                      max_length = max_length,)
        answer = self.tokenizer.batch_decode(ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)[0]
        return answer

def main():
    #csv directory
    df = pd.read_csv('resource/revised_data_courses.csv')
    db = VectorDB(API_KEY, ENVIRONMENT)

    index = db.create_index(index_name = 'course-test-index', 
                            dims = 768, 
                            metric = 'cosine')
    
    Embedder = VectorEmbedding(model_ref = 'sentence-transformers/msmarco-distilbert-base-tas-b', 
                               index = index)
    Embedder.create_embeddings(df, 'context', batch_size = 5)

    model = Model(model_checkpoint = 'google/flan-t5-large',
                  embedder = Embedder.embedder,
                  index = index)
    query = model.make_query(input('Query: '), 'context', top_k = 1)
    answer = model.generate_answer(query)

    print(answer)

if __name__ == '__main__':
    main()
