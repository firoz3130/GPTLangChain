# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 Fazza Firoz GPT Generator')
prompt = st.text_input('Typpe in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Get me important notes from youtube for the topic known as {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory
title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')
# llm = OpenAI(temperature=0.9)
# How to get something creative using the GPT-3 API key which I have
# So I will help you get something creative using the GPT-3 API key which I have
# We can develop a script so that we can fetch the data from the youtube and wikipedia/Code is below


# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
