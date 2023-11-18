import streamlit as st
import pandas as pd
import ast
from openai import OpenAI
import openai
import numpy as np
from scipy.spatial.distance import cosine

# Load the data
game_pool_df = pd.read_csv("embedded_game_info.csv")

client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key='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',
)

APP_NAME = 'Board Game Recommender'
APP_DESC = 'by @Team Siomai'

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# def search_game(df, comment, n=10, pprint=True):
#    embedding = get_embedding(comment, model='text-embedding-ada-002')
#    df['similarities'] = df['ada_embedding'].apply(lambda x: cosine(x, embedding))
#    res = df.sort_values('similarities', ascending=False).head(n)
#    return res

# Create embedding for game info
# game_pool_df['ada_embedding'] = game_pool_df['ada_embedding'].apply(ast.literal_eval)

# Embedding Function
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Similarity Score Function
def get_similarity_score(df, comment, n=3, pprint=True):
   embedding = get_embedding(comment, model='text-embedding-ada-002')
   df['similarities'] = df['ada_embedding'].apply(lambda x: cosine(eval(x), embedding))
   res_df = df.sort_values('similarities', ascending=False).head(n)
   return res_df

# Recommendation Function
def get_recommendation(target_user, comments):
    reco_prompt = f"""Synthesize the main takeaways from the following and provide recommendations for our {target_user}:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in boardgames."
            },
            {
                "role": "user",
                "content": reco_prompt + comments
            }
        ]
    )
    return response.choices[0].message['content'].strip()


# Renaming columns
game_pool_df.rename(columns={'primary': 'Game Name', 'description': 'Description', 'yearpublished': 'Year Published', 
                             'usersrated': 'Users Rated', 'average': 'Average Score'}, inplace=True)

# Page 1: Game Recommender
def get_result_1(game_description_input):
    try:
        results_1 = get_similarity_score(game_pool_df, game_description_input)
        results_1_final = results_1[['Game Name', 'Description', 'Year Published', 'Users Rated', 'Average Score']]
        results_1_final_index = results_1_final.reset_index(drop=True)
        # st.dataframe(results_1_final_index)
        return results_1_final_index
    except Exception as e  :
        print(str(e))

def get_result_2(target_user, game_title):
    try:
        results_2 = pd.read_csv("sentiment_analysis_textblob_vader.csv")
        results_2 = results_2[results_2['name'] == game_title].head()
        st.dataframe(results_2)
        # results_2 = get_recommendation(target_user, game_reviews)

    except Exception as e  :
        print(str(e))

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🇵🇭", layout="wide", initial_sidebar_state="expanded")
    st.title(APP_NAME)
    st.subheader(APP_DESC)

    target_user = st.selectbox( "Select a Target User:", ['Board Game Player', 'Board Game Designer', 'Board Game Publisher'], 0)
    game_description_input = st.text_input('Describe a board game that you would want to get recommendations from:')

    results_1 = get_result_1(game_description_input)

    game_title = st.radio("Select a game:", results_1['Game Name'])
    game_description = results_1[results_1['Game Name'] == game_title]['Description'].values[0]
    game_year_published = results_1[results_1['Game Name'] == game_title]['Year Published'].values[0]
    game_users_rated = results_1[results_1['Game Name'] == game_title]['Users Rated'].values[0]
    game_average_score = results_1[results_1['Game Name'] == game_title]['Average Score'].values[0]

    c1, c2 =  st.columns(2)
    with c1:
        st.title('Board Game Info')
        st.subheader(game_title)
        st.write(game_description)
        st.write(f"Year Published: {game_year_published}")
        st.write(f"Users Rated: {game_users_rated}")
        st.write(f"Average Score: {game_average_score}")
    with c2:
        st.title('Board Game Reviews')
        st.subheader("Top User Reviews and Sentiment Analysis")
        get_result_2(target_user, game_title)

if __name__ == "__main__":
    main()
