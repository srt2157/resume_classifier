
# Commands for execution:

For preprocessing : `python feature.py`
For Model Training and Evaluation: `python model.py`


Libraries used: Spacy, json, numpy, pandas, nltk, tensorflow.keras, sklearn

For spacy installation:
`pip install -U spacy`
`python -m spacy download en_core_web_md`

For nltk
`pip install -U nltk`

For stopwords
`python -m nltk.downloader stopwords`

Make sure you have enough GPU resources to run, for this project I used P100 on GCP.

# Assumptions

1. I'm assumed that the most recent 'experience' and 'education' are more relevant than previous history for predicting current seniority level. For eg, A SDE-2 engineer might list past internships as experience but according to recent experience he is mid-level.

2. Assumed Experience and education are sorted according to time, so I took first element for convenience.

3. From the dataset it seemed like 'skills' are processed from 'description', so ignored 'description' field.

4. I observed some fields are null, and I assumed I would pass a "empty" value as word embedding. I would like to know more on how to handle these non-numeric/non-categorical missing values in discussion session. 

5. Assumed most of the strings are in english.

# Feature Engineering

1. The dataset aldready includes following features 'title', 'skills','school','degree','work' . So I just extracted the features by parsing the data files.

2. For every feature, which most of times a string here, I did lower case conversion, special character removal and stop-word removal.

3. Ignored rows where seniority_level is none.

4. For every feature, computed feature embedding and dumped them in a file.

5.For additional features I've thought about difference b/w present and time of graduation to give an idea of seniority level, leveraging spacy to identify extra features through attributes.etc. didn't include in code.

6.Every feature has a length of 300, if you are taking 4 features into account it would be of length 4 times 300


# How I did feature analysis and trained a model?

1. For every feature, for eg, 'title' in 'experience' computed a sentence embedding of fixed size using spacy's pretrained en_core_web_md (for static embeddings). Sentence embedding is mean of individual words embeddings in that senetence. You might also use spacy's pretrained en_core_web_trf if contextualised embeddings are required.

2. For seniority.train and seniority.test, I computed embeddings and dumped them to a file.

3. The code I wrote in try1.py is flexible, one coud easily add/remove features by just editing field_subfield_map.

4. In model.py, initialised a model to a 10 layer neural network with a softmax o/p layer for this 8-class classification problem.

# Model Selection:

I choose neural network because it is giving more f1-score compared to other classifiers such as linear classifier is (~0.79).
But tradeoff is more training time.
For Neural network ~ 165.64 seconds
For linear regression ~1.8 seconds

# Results:
1. If 'title' + 'skills' is used got training accuracy of \~90% and testing accuracy \~75%, f1-score\~0.82
2. If only 'title' feature is used got training accuracy of 95%,whereas testing is 76%,f1-score\~0.82
3. If all features are used got training accuracy of \~95%,whereas testing is 75%,f1-score\~0.84


