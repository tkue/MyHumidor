import streamlit as st
import pandas as pd
import numpy as np
import csv
from streamlit_gsheets import GSheetsConnection
import gspread
import os
from getpass import getpass
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# https://stackoverflow.com/questions/76407803/define-an-output-schema-for-a-nested-json-in-langchain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
import json

from openai import OpenAI

# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = ''

# OPENAI_ORGANIZATION = getpass()
os.environ["OPENAI_ORG_ID"] = ''

class CigarItem(BaseModel):
    id: int = Field(description='read-only; do not update')
    brand: str = Field(description='Example: "Arturo Fuente"')
    series: str = Field(description='Example: "Hemmingway"')
    product: str = Field(description='The name of the specific product. This could be related to the size (e.g. "Belicoso"). Example: "Short Story"')
    msrp: float = Field(description='The Manufacturer\' Suggested Retail Price; give an answer to 2 decimal places')
    filler: str = Field(description='Example: "Dominican Republic"')
    wrapper: str = Field(description='Example: "Connecticut"')
    binder: str = Field(description='Example: "Nicaraguan"')
    ring: int = Field(description='Ring size of the cigar; Example: 56')
    length: float = Field(description='Length of the cigar; Example: 5.5')
    sweet: int = Field(description='Based on reviews and available information, give a whole number 1 to 100 that describes how sweet the cigar tastes')
    bitter: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how bitter the cigar tastes')
    creamy: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how creamy the cigar tastes')
    dry: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how dry the cigar tastes')
    salty: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how salty the cigar tastes')
    savory: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how savory the cigar tastes')
    sour: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how sour the cigar tastes')
    spicy: int = Field(description='Based on the reviews and available information, give a whole number 1 to 100 that describes how spicy the cigar tastes')
    rating_cigar_afficianado: int = Field(description='If available, give the rating from cigarafficianado.com. It should be a whole number from 1 to 100.')
    rating_halfwheel: int = Field(description='If available, give the rating from halfwheel.com. It should be a whole number from 1 to 100.')


class CigarList(BaseModel):
    cigars: List[CigarItem]


def get_gsheets_dataframe():
    gc = gspread.service_account(filename='credentials.json')
    sh = gc.open_by_key('1VnTK0xg9NyxOwITvcQH8a5WwHJzJogy4W7ahRhh4kBc')
    ws = sh.sheet1

    res = ws.get_all_records()
    return pd.DataFrame(res)


df = get_gsheets_dataframe()
df.set_index('ID')


"""sample
{
  "cigars": [
    {
      "id": 13,
      "brand": "Arturo Fuente",
      "series": "Hemingway",
      "msrp": 7.25,
      "filler": "Dominican Republic",
      "wrapper": "Cameroon",
      "binder": "Dominican Republic",
      "ring": 49,
      "length": 4.0,
      "sweet": 70,
      "bitter": 30,
      "creamy": 60,
      "dry": 25,
      "salty": 5,
      "savory": 20,
      "sour": 10,
      "spicy": 40,
      "rating_cigar_afficianado": 89,
      "rating_halfwheel": 91
    }
  ]
}

"""


# exit()


sample_input = df.iloc[[11]].to_json(orient='records')
in_parser = PydanticOutputParser(pydantic_object=CigarList)



client = OpenAI()
client.api_key = os.environ.get('OPENAI_API_KEY')
completion = client.chat.completions.create(
    model="gpt-4-turbo",
    response_format={"type": "json_object"},
    temperature=0,
    messages=[
        {"role": "system",
         "content": f"You are providing information about cigars and need to be as accurate as possible. You are given a JSON array and must return a JSON object with the fields filled in. Format instructions for the return object: {in_parser.get_format_instructions()}"},
        {"role": "user",
         "content": f"Fill in the missing information in the following JSON array. Return a JSON object. This is in context of cigars. {sample_input}"}
    ]
)

print(completion.choices[0].message.content)

"""



out_parser = JsonOutputToolsParser()

# Set up LLM
model = ChatOpenAI(model="gpt-4-turbo", temperature=0.8).bind_tools([CigarList, CigarItem], max_tokens=4000)
prompt = ChatPromptTemplate.from_messages(
    [('system', 'You are providing information about cigars and need to be as accurate as possible. You are given a JSON array and must return a JSON object with the fields filled in'), ('user', '{input}')]
)


model.kwargs['tools'][0]['function']['description'] = model.kwargs['tools'][0]['function']['description'][:1024]
model.kwargs['tools'][1]['function']['description'] = model.kwargs['tools'][1]['function']['description'][:1024]

chain = prompt | model | out_parser
# chain.max_tokens_limit = 4000

# print(len(str(model.kwargs['tools'])))

# print(sample_input)
print(chain.invoke({'input': sample_input}))
"""

# llm = OpenAI(openai_api_key=os.environ.get('OPEN_API_KEY'), openai_organization=os.environ.get('OPENAI_ORG_ID'), temperature=.8, model_name="gpt-3.5-turbo-0125")
#
# # Set up memory
# memory = ConversationBufferMemory(return_messages=True)
#
# # Set up the prompt
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(system_message),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("""{input}""")
# ])
#
# # Create conversation chain
# conversation = ConversationChain(memory=memory, prompt=prompt,
#                                 llm=llm, verbose=False)


# st.table(df)