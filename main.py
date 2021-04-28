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

import pandas as pd
import numpy as np
import tensorflow as tf

# variables to configure
proportion = 0.1  # proportion of dataset to use for training
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
    "left": 0.125,
    "left-center": 0.125,
    "center": 0.5,
    "allsides": 0,
    "right-center": 0.125,
    "right": 0.125
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


def preprocess(proportion, scoring, distribution):
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

    for index, row in df.sample(frac=proportion).iterrows():
        print(index, row)
        publication = row['publication']
        content = row['content']
        print(publication, content)
        try:
            # check alignment of article
            selection = allsides.loc[allsides["name"] == publication]
            alignment = selection["bias"].values[0]

            # check if it falls within desired distribution
            if count[alignment] < items * distribution[alignment]:
                score = scoring[alignment]
                scorearray.append(score)
                contentarray.append(content)

        except KeyError:
            print("Warning: Publication", publication, "not in dataset")
    df_processed = pd.DataFrame(zip(scorearray, contentarray), columns=["score", "content"])
    # proportion based
    print(df_processed)
    return df_processed


def train(df):
    print("Initializing")


# run
if __name__ == '__main__':
    data = preprocess(proportion, scoring, distribution)
    train(data)
