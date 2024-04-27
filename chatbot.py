import numpy as np
import pandas as pd
import random 
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

QAData = 'Data/QADataSet.csv'
smallTalkData = 'Data/smalltalk.csv'
intentData = 'Data/intent.csv'
nbaQAData = 'Data/NBAQAtest.csv'

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def filter_name(tokenized_name):
    tokens = word_tokenize(tokenized_name)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ''.join(filtered_tokens)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    tfidf_vectorized = tfidf_vectorizer.fit_transform(data).toarray()
    return tfidf_vectorized, tfidf_vectorizer

def find_response(query, data, tfidf_vectorizer, threshold):
    tfidf_query = tfidf_vectorizer.transform([preprocess(query)]).toarray()
    cosine_sim = cosine_similarity(data, tfidf_query)
    max_sim = cosine_sim.max()
    if max_sim >= threshold:
        max_idx = np.where(cosine_sim == max_sim)[0][0]
        return data.iloc[max_idx]['Response']
    else:
        return 'noQuery'

def get_time():
    return f"The time right now is {time.strftime('%H:%M', time.localtime())}"

def get_date():
    return f"The date right now is {time.strftime('%A %d %B %Y', time.localtime())}"

def instantiate_bot():
    chat_bot = "LeBron"
    user_name = "Guest"

    print(f"{chat_bot}: Hello, I am LeBron James, your general chatbot.")

    query_name = input(f"{chat_bot}: What is your name?\n")
    if query_name:
        user_name = filter_name(query_name)
        print(f"[Your user name has been set to {user_name}]")
    else:
        print(f"[Your user name has been set to {user_name}]")

    print(f"{chat_bot}: Hello, {user_name}, how are you? :)")

    while True:
        query = input(f"{user_name}: ")
        if query.lower() == 'bye':
            print(f"{chat_bot}: Bye {user_name}, see you next time!")
            break

        small_talk_response = find_response(query, small_talk_tfidf, small_talk_vectorizer, 0.8)
        intent = find_response(query, intent_data, intent_vectorizer, 0.8)
        qa_answer = find_response(query, qa_data, qa_vectorizer, 0.9)
        nba_answer = find_response(query, nba_qa_data, nba_vectorizer, 0.5)

        if intent == 'change_name':
            new_name = input(f"{chat_bot}: What would you like to change your name to?\n")
            user_name = filter_name(new_name) if new_name else "Guest"
            print(f"[Success! Your user name has been set to {user_name}]")
        elif intent == 'get_time':
            print(f"{chat_bot}: {get_time()}")
        elif intent == 'get_date':
            print(f"{chat_bot}: {get_date()}")
        elif intent == 'say_bye':
            print(f"{chat_bot}: Bye {user_name}, see you next time!")
            break
        elif small_talk_response != 'noQuery':
            print(f"{chat_bot}: {small_talk_response}")
        elif qa_answer != 'noQuery':
            print(f"{chat_bot}: {qa_answer}")
        elif nba_answer != 'noQuery':
            print(f"{chat_bot}: {nba_answer}")
        else:
            print(f"{chat_bot}: I did not understand :(, please say something else")

if __name__ == "__main__":
    small_talk_data = load_data(smallTalkData)
    small_talk_tfidf, small_talk_vectorizer = preprocess_data(small_talk_data['Utterances'])

    intent_data = load_data(intentData)
    intent_tfidf, intent_vectorizer = preprocess_data(intent_data['Utterance'])

    qa_data = load_data(QAData)
    qa_tfidf, qa_vectorizer = preprocess_data(qa_data['Question'])

    nba_qa_data = load_data(nbaQAData)
    nba_tfidf, nba_vectorizer = preprocess_data(nba_qa_data['Question'])

    instantiate_bot()
