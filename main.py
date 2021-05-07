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
#   15 different American news sources
#   143,000 unique articles
#   32,000 randomly selected articles at a time

import json
import pandas as pd
import nltk
import numpy as np
import re
# import tensorflow as tf

# variables to configure
# proportion = 0.1  # proportion of dataset to use for training
cutoff = 10000   # use x most occurring words in bag
articles = 32000   # total number of articles to use in training
margin = 0.25
scoring = {  # bias rating to give toward each alignment
    #   algorithm in use: left, right, left-center, right-center = +1 | center or allsides = -1
    #   alternate option: left -2, leftcenter -1, allsides/center +0, right-center +1, right +2
    "left": -2,
    "left-center": -1,
    "center": 0,
    "allsides": 0,
    "right-center": 1,
    "right": 2
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
    print("Initializing data processing")
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

    df_raw = df.sample(frac=1)
    for index, row in df_raw.iterrows():
        publication = row['publication']
        content = row['content']
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
                if row["id"] % 100 == 0:
                    print(100*sum(count.values())/sum(distribution.values()), "%", sep="")
                    if not distribute(items):
                        break
        except KeyError:
            print("Warning: Publication", publication, "not in dataset")
    df_processed = pd.DataFrame(zip(scorearray, contentarray), columns=["score", "content"])
    print(count)
    return df_processed


def bagify(df):
    print("Initializing bag algorithm")
    bag = {}
    length = len(df.index)
    # with open("DATA/bagofwords_lean.json") as file:
    #    bag = json.load(file)
    for index, row in df[["score", "content"]].iterrows():
        if index % 100 == 0:
            print(index/length*100, "%", sep="")
        score = row["score"]
        content = row["content"]
        wordslist = nltk.word_tokenize(content)
        # usedwords array to delete duplicates from same article
        usedwords = []
        for word in wordslist:
            word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(word) <= 1 or word in usedwords:
                continue
            elif word in bag:
                bag[word]["count"] += 1
                bag[word]["value"] += score
            else:
                bag[word] = {"count": 1, "value": score}
            usedwords.append(word)

    # filter rare usage counts
    original_length = len(bag)
    bag = {k: v for k, v in reversed(sorted(bag.items(), key=lambda item: item[1]["count"])[-cutoff:])}
    print("Saved", cutoff, "most common words out of", original_length, "total words")

    with open("DATA/bagofwords_lean.json", "w") as file:
        json.dump(bag, file)
    return bag


def train(df):
    # find weights
    with open("DATA/bagofwords_lean.json") as file:
        bag = json.load(file)
    model = 0
    return model


def analyze():
    # data insights from saved bag
    with open("DATA/bagofwords_lean.json") as file:
        bag = json.load(file)
    print("50 most common words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["count"])[-50:])})
    print("50 most uncommon words:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["count"])[:50]})
    print("50 most biased words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"])[-50:])})
    print("50 most unbiased words:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"])[:50]})
    print("50 most bias per usage", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[-50:])})
    print("50 most unbiased per usage:", {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[:50]})
    # print("50 most weighed words:", {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["weight"])[-50:])})


def analyze_list():
    # data insights from saved bag
    with open("DATA/bagofwords_lean.json") as file:
        bag = json.load(file)
    common_words = {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["count"])[-50:])}
    uncommon_words = {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["count"])[:50]}
    biased = {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"])[-50:])}
    unbiased = {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"])[:50]}
    bpu = {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[-50:])}
    ubpu = {k: v for k, v in sorted(bag.items(), key=lambda word: word[1]["value"]/word[1]["count"])[:50]}

    print("\n50 Most Common words: ")
    for pair in common_words.items():
        print(pair)

    print("\n50 Most Uncommon words: ")
    for pair in uncommon_words.items():
        print(pair)

    print("\n50 Most Biased words: ")
    for pair in biased.items():
        print(pair)

    print("\n50 Most Unbiased words: ")
    for pair in unbiased.items():
        print(pair)


def predict(article):
    with open("DATA/bagofwords_lean.json") as file:
        bag = json.load(file)
    bagged_words = {k: v for k, v in bag.items()}
    # common = {k: v for k, v in reversed(sorted(bag.items(), key=lambda word: word[1]["count"])[-50:])}.keys()
    value = 0
    count = 0
    wordslist = nltk.word_tokenize(article)
    usedwords = []
    for word in wordslist:
        word = re.sub(r'[^a-zA-Z]', '', word).lower()
        if word in bagged_words.keys() and word not in usedwords:  # and word not in common:
            score = bag[word]["value"] / bag[word]["count"]
            value += score
            count += 1
            usedwords.append(word)
    print("Total score:", value)
    print("Words analyzed:", count)
    try:
        result = value / count
        print("Bias score:", result)
        if abs(result) > margin:
            print("Biased article:")
            if result > 0:
                print("Right-leaning")
            else:
                print("Left-leaning")
        else:
            print("Unbiased article")
        return result
    except ArithmeticError:
        print("Unable to determine bias because no words were analyzed.")
        return -3


def test(df, length=None):
    print("Initializing test")
    if length is None:
        length = len(df.index)
    correct = 0
    total = 0
    sample = df[["score", "content"]].sample(frac=1)
    for index, row in sample.iterrows():
        score = row["score"]
        content = row["content"]
        prediction = predict(content)
        if (abs(prediction) > margin and 0 < abs(score) <= 2) or (abs(prediction) < margin and score == 0):
            correct += 1
            total += 1
        else:
            total += 1
        if total >= length:
            break
    print("Accuracy: ", correct/total*100, "%", sep="")


if __name__ == '__main__':
    # data = preprocess(scoring, distribution)
    # data.to_csv("DATA/sample_lean.csv", index=True)
    data = pd.read_csv("DATA/sample_lean.csv")
    # bagofwords = bagify(data)

    # analyze_list()
    test(data, 100)
    analyze()
    # with open("article_to_predict.txt", encoding="utf8") as file:
    #    predict_text = file.read().replace("\n", " ")
    # predict(predict_text)
