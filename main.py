# Almutwakel Hassan

# Bias analysis
# Process: Look at articles with assigned bias indicators to create a bag of words,
# the dataset should have an equal number of biased (-) and unbiased (+) scores
# (so that common words don't have swayed ratings),
# each word is trained to have its own bias rating (negative is bias, positive is unbias)
# After training is over, remove words that don't have at least 100 references in the data
# Save the model for reuse
# This bag of words model is applied to an inputted text and a bias rating is produced

# This software has limitations; bias is dependent on the meaning of the words and not just the words used
# However this algorithm still provides a useful estimation to bias
# It is the first step on my quest to challenge the spread of misinformation

def main():
    #
    print("Initializing")


# run
if __name__ == '__main__':
    main()

