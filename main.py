# Almutwakel Hassan

# Bias analysis
# Process: Large article dataset is used to create a bag of words,
# bias indicators are assigned to each article.
# Bias scores are assigned based on publication and come from AllSides Bias Rating dataset.
# the dataset should have an equal number of biased (+) and unbiased (-) scores
# (so that common words don't have swayed ratings),
# each word is trained to have its own bias rating (positive is bias, positive is bias)
# After training is over, remove words that don't have at least 100 references in the data
# Save the model for reuse
# This bag of words model is applied to an inputted text and a bias rating is produced

# This software has limitations; bias is dependent on the meaning of the words and not just the words used
# However this algorithm still provides a useful estimation to bias
# It is the first step on my quest to challenge the spread of misinformation

# Article Dataset Information:
#   639 MB of data
#   143,000 unique articles
#   15 different American news sources

import json
import pandas as pd
import nltk
import numpy as np
import re
import tensorflow as tf

# variables to configure
# proportion = 0.1  # proportion of dataset to use for training
cutoff = 10000   # use x most occurring words in bag
articles = 32000   # total number of articles to use in training
scoring = {  # bias rating to give toward each alignment
    #   algorithm in use: left, right, left-center, right-center = +1 | center or allsides = -1
    #   alternate option: left -2, leftcenter -1, allsides/center +0, right-center +1, right +2
    "left": 1,
    "left-center": 1,
    "center": -1,
    "allsides": -1,
    "right-center": 1,
    "right": 1
}
distribution = {  # the distribution of each alignment used in the dataset
    #   algorithm in use: 1:1:4:0:1:1 ratio
    "left": articles * 0.125,
    "left-center": articles * 0.125,
    "center": articles * 0.5,
    "allsides": articles * 0,
    "right-center": articles * 0.125,
    "right": articles * 0.125
}
# do not configure
count = {
    "left": 0,
    "left-center": 0,
    "center": 0,
    "allsides": 0,
    "right-center": 0,
    "right": 0
}
delchars = ''.join(c for c in map(chr, range(256)) if not c.isalpha())


def distribute(amount=1):
    for alignment in ["left", "left-center", "center", "allsides", "right-center", "right"]:
        if count[alignment] < distribution[alignment]:   # * amount * proportion
            return True
    return False


def preprocess(scoring, distribution):
    allsides = pd.read_csv("DATA/allsides.csv", sep=",")[["name", "bias"]]
    # print(np.count_nonzero(allsides["bias"] == "right-center")/3)

    # load in data
    data1 = pd.read_csv("DATA/articles1.csv", sep=",")[["id", "publication", "content"]]
    data2 = pd.read_csv("DATA/articles2.csv", sep=",")[["id", "publication", "content"]]
    data3 = pd.read_csv("DATA/articles3.csv", sep=",")[["id", "publication", "content"]]

    # concatenate columns
    df = pd.concat([data1, data2, data3], ignore_index=True)

    # transform data
    items = len(df)
    scorearray = []
    contentarray = []

    for index, row in df.sample(frac=1).iterrows():
        # for index, row in df.groupby("score").apply(lambda x: x.sample(n=0.1*items)):
        # print(index, row)
        publication = row['publication']
        content = row['content']
        # while distribute(items):
        # row = df.loc[i]
        # publication = row['publication']
        # content = row['content']
        # print(publication, content)
        try:
            # check alignment of article
            selection = allsides.loc[allsides["name"] == publication]
            alignment = selection["bias"].values[0]

            # check if it falls within desired distribution
            if count[alignment] < distribution[alignment]:  # * items * proportion
                score = scoring[alignment]
                scorearray.append(score)
                contentarray.append(content)
                count[alignment] += 1
                if row["id"] % 10 == 0 and not distribute(items):
                    break
        except KeyError:
            print("Warning: Publication", publication, "not in dataset")
    df_processed = pd.DataFrame(zip(scorearray, contentarray), columns=["score", "content"])
    print(count)
    # print(df_processed)
    print(np.count_nonzero(df_processed["score"] == 1), np.count_nonzero(df_processed["score"] == -1))
    return df_processed


def bagify(df):
    print("Initializing")
    bag = {}
    # with open("DATA/bagofwords_sample.json") as file:
    #    bag = json.load(file)
    for index, row in df[["score", "content"]].iterrows():
        score = row["score"]
        content = row["content"]
        wordslist = nltk.word_tokenize(content)
        # usedwords array to delete duplicates from same article
        usedwords = []
        for word in wordslist:
            word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(word) <= 1 or word in usedwords:
                pass
            elif word in bag:
                bag[word]["count"] += 1
                bag[word]["value"] += score
            else:
                bag[word] = {"count": 1, "value": score, "weight": 1}
            usedwords.append(word)

    # filter rare usage counts
    original_length = len(bag)
    bag = {k: v for k, v in reversed(sorted(bag.items(), key=lambda item: item[1]["count"])[-cutoff:])}
    print("Saved", cutoff, "most common words out of", original_length, "total words")

    with open("DATA/bagofwords.json", "w") as file:
        json.dump(bag, file)
    return bag


def train():
    # find weights
    model = 0
    return model


def analyze():
    # data insights from saved bag
    with open("DATA/bagofwords.json") as file:
        bag = json.load(file)
    print("50 most common words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["count"])[-50:])})
    print("50 most uncommon words:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["count"])[:50]})
    print("50 most biased words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"])[-50:])})
    print("50 most unbiased words:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"])[:50]})
    print("50 most bias per usage", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[-50:])})
    print("50 most unbiased per usage:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[:50]})
    print("50 most weighted words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["weight"])[-50:])})


def predict(article):
    with open("DATA/bagofwords.json") as file:
        bag = json.load(file)
    bagged_words = {k: v for k, v in bag.items()}
    value = 0
    count = 0
    wordslist = nltk.word_tokenize(article)
    usedwords = []
    for word in wordslist:
        # print(word)
        word = re.sub(r'[^a-zA-Z]', '', word).lower()
        if word in bagged_words.keys() and word not in usedwords and word not in {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["count"])[-50:])}.keys():
            value += bag[word]["value"] * bag[word]["weight"] / bag[word]["count"]
            count += 1
            usedwords.append(word)
    print("Total score:", value)
    print("Words analyzed:", count)
    try:
        print("Final bias score:", value/count)
    except ArithmeticError:
        print("Unable to determine bias because no words were analyzed.")


# run
if __name__ == '__main__':
    # data = preprocess(scoring, distribution)
    # bagofwords = bagify(data)
    # can use saved data for analysis and training:
    analyze()
    # train()
    with open("article_to_predict.txt", encoding="utf8") as file:
        predict_text = file.read().replace("\n", " ")
    # predict(predict_text)
