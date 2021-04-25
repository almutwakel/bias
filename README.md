# bias-ai
Uses machine learning to determine bias in a text input

# Almutwakel Hassan

Bias analysis
Process: Large article dataset is used to create a bag of words,
bias indicators are assigned to each article.
Bias scores are assigned based on publication and come from AllSides Bias Rating dataset.
the dataset should have an equal number of biased (+) and unbiased (-) scores
(so that common words don't have swayed ratings),
each word is trained to have its own bias rating (positive is bias, positive is bias)
After training is over, remove words that don't have at least 100 references in the data
Save the model for reuse
This bag of words model is applied to an inputted text and a bias rating is produced

This software has limitations; bias is dependent on the meaning of the words and not just the words used
However this algorithm still provides a useful estimation to bias
It is the first step on my quest to challenge the spread of misinformation

Article Dataset Information:
  639 MB of data
  143,000 unique articles
  15 different American news sources
