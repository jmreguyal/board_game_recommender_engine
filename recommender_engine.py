import streamlit as st
import pandas as pd
import ast
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine

# Load the data
game_pool_df = pd.read_csv("embedded_game_info.csv")

client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key='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',
)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_game(df, comment, n=10, pprint=True):
   embedding = get_embedding(comment, model='text-embedding-ada-002')
   df['similarities'] = df['ada_embedding'].apply(lambda x: cosine(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

# Create embedding for game info
game_pool_df['ada_embedding'] = game_pool_df['ada_embedding'].apply(ast.literal_eval)

# Renaming columns
game_pool_df.rename(columns={'primary': 'Game Name', 'description': 'Description', 'yearpublished': 'Year Published', 
                   'usersrated': 'Users Rated', 'average': 'Average Score'}, inplace=True)

# Page 1: Game Recommender
def page1():
    # st.image('board_game_logo_2.png', width = 600)
    st.title('Board Game Recommender')
    game_description_input = st.text_input('Input board game description you want to get recommendations from:')
    try:
        results = search_game(game_pool_df, game_description_input)
        results_final = results[['Game Name', 'Description', 'Year Published', 'Users Rated', 'Average Score']]
        results_final_index = results_final.reset_index(drop=True)
        st.dataframe(results_final_index, width = 700)
    except:
        st.error("Please input a valid description!")

def main():
    # page = st.sidebar.radio("Choose a page:", ("Recommendation Engine - SB19 to Top 200", "Recommendation Engine - Top200 to SB19", "Recommendation Engine - Audio Features to SB19"))
    page1()

if __name__ == "__main__":
    main()
