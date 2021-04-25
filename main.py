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

# load in data
data1 = pd.read_csv("DATA/articles1.csv", sep=",")[["id", "publication", "content"]]
data2 = pd.read_csv("DATA/articles2.csv", sep=",")[["id", "publication", "content"]]
data3 = pd.read_csv("DATA/articles3.csv", sep=",")[["id", "publication", "content"]]

# concatenate columns
idarray = data1["id"].append(data2["id"]).append(data3["id"])
publicationarray = data1["publication"].append(data2["publication"]).append(data3["publication"])
contentarray = data1["content"].append(data2["content"]).append(data3["content"])

# create pandas dataframe
df = pd.DataFrame(zip(idarray, publicationarray, contentarray), columns=['id', 'publication', 'content'])

print(df)
# transform data into numbers
# left, right, left-center, right-center = +1 | center or allsides = -1


def main():
    print("Initializing")


# run
if __name__ == '__main__':
    main()

