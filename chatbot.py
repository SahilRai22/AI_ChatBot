import numpy
import numpy 
import random 
import time
import pandas 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk.tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

QAData = 'Data/QADataSet.csv'
smallTalkData = 'Data/smalltalk.csv'
intentData = 'Data/intent.csv'
nbaQAData = 'Data/NBAQAtest.csv'


def main():

    chatBot = "LeBron"
    userName = "Guest"
    lemmatiser = WordNetLemmatizer()

    def preProcess(text):
        lemmatiser = WordNetLemmatizer()
        textTokens = word_tokenize(text)
        tokens = [word.lower() for word in textTokens if not word in stopwords.words('english')]
        tokens = lemmatiser.lemmatize(tokens)
        return (''.join(tokens))

    def filterName(tokenizeName):
        nameToken = word_tokenize(tokenizeName)
        removeStopWord = [word for word in nameToken if not word in stopwords.words()]
        # post = nltk.pos_tag(removeStopWord, tagset='universal')
        name = ''.join(''.join(tup) for tup in removeStopWord)
        return name
    
    # def filterName(tokenizeName):
    #     nameToken = word_tokenize(tokenizeName)
    #     removeStopWord = [word for word in nameToken if not word in stopwords.words()]
    #     # post = nltk.pos_tag(removeStopWord, tagset='universal')
    #     name = ''.join(''.join(tup) for tup in removeStopWord)
    #     return nam

    def startSmallTalk(query, threshold):
        stDB = pandas.read_csv(smallTalkData) 
        tfdifVectorize = TfidfVectorizer(analyzer='word')
        tfidfVectorized = tfdifVectorize.fit_transform(stDB['Utterances']).toarray()
        dataframeTfidf = pandas.DataFrame(tfidfVectorized, columns = tfdifVectorize.get_feature_names_out())
        putTfid = tfdifVectorize.transform([query.lower()]).toarray()
        cosineSim = 1- pairwise_distances(dataframeTfidf, putTfid, metric= 'cosine')
        if threshold <= cosineSim.max():
            id_argmax = numpy.where(cosineSim == numpy.max(cosineSim, axis = 0))
            id = numpy.random.choice(id_argmax[0])
            return stDB['Reponse'].loc[id]
        else:
            return 'noQuery'

    def startNBAQA(query, threshold):
        qaDB = pandas.read_csv(nbaQAData)
        tfdifVectorize = TfidfVectorizer(analyzer='word')
        tfidfVectorized = tfdifVectorize.fit_transform(qaDB['Question']).toarray()
        dataframeTfidf = pandas.DataFrame(tfidfVectorized, columns = tfdifVectorize.get_feature_names_out())
        putTfid = tfdifVectorize.transform([query.lower()]).toarray()
        cosineSim = 1- pairwise_distances(dataframeTfidf, putTfid, metric= 'cosine')

        if threshold <= cosineSim.max():
            id_argmax = numpy.where(cosineSim == numpy.max(cosineSim, axis = 0))
            id = numpy.random.choice(id_argmax[0])
            return qaDB['Answer'].loc[id]
        else:
            return 'noQuery'

    def startQA(query, threshold):
        qaDB = pandas.read_csv(QAData)
        tfdifVectorize = TfidfVectorizer(analyzer='word')
        tfidfVectorized = tfdifVectorize.fit_transform(qaDB['Question']).toarray()
        dataframeTfidf = pandas.DataFrame(tfidfVectorized, columns = tfdifVectorize.get_feature_names_out())
        putTfid = tfdifVectorize.transform([query.lower()]).toarray()
        cosineSim = 1- pairwise_distances(dataframeTfidf, putTfid, metric= 'cosine')

        if threshold <= cosineSim.max():
            id_argmax = numpy.where(cosineSim == numpy.max(cosineSim, axis = 0))
            id = numpy.random.choice(id_argmax[0])
            return qaDB['Answer'].loc[id]
        else:
            return 'noQuery'

    def startIntent(query, threshold):
        iDB = pandas.read_csv(intentData)
        tfdifVectorize = TfidfVectorizer(analyzer='word')
        tfidfVectorized = tfdifVectorize.fit_transform(iDB['Utterance']).toarray()
        dataframeTfidf = pandas.DataFrame(tfidfVectorized, columns = tfdifVectorize.get_feature_names_out())
        putTfid = tfdifVectorize.transform([query.lower()]).toarray()
        cosineSim = 1- pairwise_distances(dataframeTfidf, putTfid, metric= 'cosine')

        if threshold <= cosineSim.max():
            id_argmax = numpy.where(cosineSim == numpy.max(cosineSim, axis = 0))
            id = numpy.random.choice(id_argmax[0])
            return iDB['Intent'].loc[id]

    def getTime():
        responses = ["The time right now is ", "The time is ", "Currently it is "]

        clock = time.time()
        timeResponse = random.choice(responses) + time.strftime('%H:%M', time.localtime(clock))
        return timeResponse

    def getDate():
        responses = ["Tue date right now is ", "The date is ", "Today's date is "]
        clock = time.time()
        dataResponse = random.choice(responses) + time.strftime('%A %d %B %Y', time.localtime(clock))
        return dataResponse
     
    def instantiateBot():
        stopQuery = False 
        # ---- Name management ---- 
        queryName = input(chatBot + ": Hello I am Lebron James, your general chatbot what is your name?\n")
        if(queryName):
                userName = filterName(queryName)
                print(f"[Your user name has been set to {userName}]")
                time.sleep(1.5)
                print("[note you can change your name, Just ask LeBron!] ")
                time.sleep(1.5)
  
        else:
            userName = "Guest"
            #time.sleep(1.5)
            print(f"{chatBot}: Your user name has been set to {userName}")
            time.sleep(1.5)
            print(">> note you can change your name, Just ask LeBron! :) ")
            time.sleep(1.5)
            #print(f"{chatBot}: Hello ", userName)
               

        print(f"{chatBot}: Hello, {userName} how are you? :)")
        while not stopQuery:
            query = input(f"{userName}: ")
            smallTalkResponse = startSmallTalk(query, threshold = 0.8)
            intents = startIntent(query, threshold = 0.8)
            qaAnswers = startQA(query, threshold = 0.9)
            nbaAnswers = startNBAQA(query, threshold = 0.5)
            # ---- Resolve Query ----  

            if  intents == 'change_name':
                queryName = input(chatBot + ": What would you like to change your name to?\n")
                userName = filterName(queryName)
                if userName == '':
                    userName = "Guest"
                print(f"[Success!! Your user name has been set to {userName}]")
                print(f"{chatBot}: Hello, {userName} could i help you with anything else?")
                continue  

            elif intents == 'get_time':
                print(f"{chatBot}:" , getTime())

            elif intents == 'get_date':
                print(f"{chatBot}:" , getDate())
            
            elif intents == 'say_bye':
                print(f"{chatBot}: Bye {userName}, see you next time!")
                stopQuery = True
                break

            elif smallTalkResponse !='noQuery':
                print(f"{chatBot}:", smallTalkResponse)
                continue

            elif qaAnswers != 'noQuery':
                 print(f"{chatBot}:", qaAnswers)
                 print(f"{chatBot}:", "Could I help you with anything else?")

            elif nbaAnswers != 'noQuery':
                print(f"{chatBot}:", nbaAnswers)
                print(f"{chatBot}:", "Could I help you with anything else?")

            else:
                print(f"{chatBot}: I did not understand :(, please say something else " )

    instantiateBot()
main()      
    


