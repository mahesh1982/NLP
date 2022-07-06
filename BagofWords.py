# define the data to be used
dataDict = {
    "sentence":[
        "Avengers is a great movie.",
        "I love Avengers it is great.",
        "Avengers is a bad movie.",
        "I hate Avengers.",
        "I didnt like the Avengers movie.",
        "I think Avengers is a bad movie.",
        "I love the movie.",
        "I think it is great."
    ],
    "sentiment":[
        "good",
        "good",
        "bad",
        "bad",
        "bad",
        "bad",
        "good",
        "good"
    ]
}
# define a list of stopwords
stopWrds = ["is", "a", "i", "it"]
# define model training parameters
epochs = 30
batchSize = 10
# define number of dense units
denseUnits = 50
#######################################################################################################
# import the necessary packages
import re


def preprocess(sentDf, stopWords, key="sentence"):
    # loop over all the sentences
    for num in range(len(sentDf[key])):
        # lowercase the string and remove punctuation
        sentence = sentDf[key][num]
        sentence = re.sub(
            r"[^a-zA-Z0-9]", " ", sentence.lower()
        ).split()
        # define a list for processed words
        newWords = list()
        # loop over the words in each sentence and filter out the
        # stopwords
        for word in sentence:
            if word not in stopWords:
                # append word if not a stopword
                newWords.append(word)
        # replace sentence with the list of new words
        sentDf[key][num] = newWords

    # return the preprocessed data
    return sentDf


def prepare_tokenizer(df, sentKey="sentence", outputKey="sentiment"):
    # counters for tokenizer indices
    wordCounter = 0
    labelCounter = 0
    # create placeholder dictionaries for tokenizer
    textDict = dict()
    labelDict = dict()
    # loop over the sentences
    for entry in df[sentKey]:
        # loop over each word and
        # check if encountered before
        for word in entry:
            if word not in textDict.keys():
                textDict[word] = wordCounter
                # update word counter if new
                # word is encountered
                wordCounter += 1

    # repeat same process for labels
    for label in df[outputKey]:
        if label not in labelDict.keys():
            labelDict[label] = labelCounter
            labelCounter += 1

    # return the dictionaries
    return (textDict, labelDict)

def calculate_bag_of_words(text, sentence):
    # create a dictionary for frequency check
    freqDict = dict.fromkeys(text, 0)
    # loop over the words in sentences
    for word in sentence:
        # update word frequency
        freqDict[word]=sentence.count(word)
    # return dictionary
    return freqDict

#######################################################################################################
# import the necessary packages
from tensorflow.keras.preprocessing.text import Tokenizer
def tensorflow_wrap(df):
    # create the tokenizer for sentences
    tokenizerSentence = Tokenizer()
    # create the tokenizer for labels
    tokenizerLabel = Tokenizer()
    # fit the tokenizer on the documents
    tokenizerSentence.fit_on_texts(df["sentence"])
    # fit the tokenizer on the labels
    tokenizerLabel.fit_on_texts(df["sentiment"])
    # create vectors using tensorflow
    encodedData = tokenizerSentence.texts_to_matrix(
        texts=df["sentence"], mode="count")
    # add label column
    labels = df["sentiment"]
    # correct label vectors
    for i in range(len(labels)):
        labels[i] = tokenizerLabel.word_index[labels[i]] - 1
    # return data and labels
    return (encodedData[:, 1:], labels.astype("float32"))

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
def build_shallow_net():
    # define the model
    model = Sequential()
    model.add(Dense(denseUnits, input_dim=10, activation="relu"))
    model.add(Dense(denseUnits, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # compile the keras model
    model.compile(loss="binary_crossentropy", optimizer="adam",
        metrics=["accuracy"]
    )
    # return model
    return model
##################################################################################
import pandas as pd
# convert the input data dictionary to a pandas data frame
df = pd.DataFrame.from_dict(dataDict)
# preprocess the data frame and create data dictionaries
preprocessedDf = preprocess(sentDf=df, stopWords=stopWrds)
(textDict, labelDict) = prepare_tokenizer(df)
# create an empty list for vectors
freqList = list()
# build vectors from the sentences
for sentence in df["sentence"]:
    # create entries for each sentence and update the vector list
    entryFreq = calculate_bag_of_words(text=textDict,
        sentence=sentence)
    freqList.append(entryFreq)

# create an empty data frame for the vectors
finalDf = pd.DataFrame()
# loop over the vectors and concat them
for vector in freqList:
    vector = pd.DataFrame([vector])
    finalDf = pd.concat([finalDf, vector], ignore_index=True)
# add label column to the final data frame
finalDf["label"] = df["sentiment"]
# convert label into corresponding vector
for i in range(len(finalDf["label"])):
    finalDf["label"][i] = labelDict[finalDf["label"][i]]
# initialize the vanilla model
shallowModel = build_shallow_net()
print("[Info] Compiling model...")
# fit the Keras model on the dataset
shallowModel.fit(
    finalDf.iloc[:,0:10],
    finalDf.iloc[:,10].astype("float32"),
    epochs=epochs,
    batch_size=batchSize
)

# create dataset using TensorFlow
trainX, trainY = tensorflow_wrap(df)
# initialize the new model for tf wrapped data
tensorflowModel = build_shallow_net()
print("[Info] Compiling model with tensorflow wrapped data...")
# fit the keras model on the tensorflow dataset
tensorflowModel.fit(
    trainX,
    trainY,
    epochs=epochs,
    batch_size=batchSize
)