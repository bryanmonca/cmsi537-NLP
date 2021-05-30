import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model  import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

## Importing data
# Train set
train_sents = []
train_labels = []
with open('/Users/bryanmonca/Files/LMU/CMSI_537_FALL/Assignments/ps0-bryanmonca/sentiment-data/train.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        train_sents.append(line[1:])
        train_labels.append(int(line[0]))

# Val set
val_sents = []
val_labels = []
with open('/Users/bryanmonca/Files/LMU/CMSI_537_FALL/Assignments/ps0-bryanmonca/sentiment-data/val.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        val_sents.append(line[1:])
        val_labels.append(int(line[0]))

## Baseline classifier
prediction = 1

# Scores
y_train = np.array(train_labels)
scores = np.where(prediction == y_train, 1, 0)
bl_train_score = np.mean(scores)
print("Base line train set accuracy: ", bl_train_score) # 52.17%

y_val = np.array(val_labels)
scores = np.where(prediction == y_val, 1, 0)
bl_val_score = np.mean(scores)
print("Base line val set accuracy: ", bl_val_score) # 50.92%

## Vectorizing
# Train set vectorizer
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train)
X_train = X_train.toarray()

# Val set vectorizer
val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer.transform(val)
X_val = X_val.toarray()


## Multinomial Naive Bayes classifier
model_nb  = MultinomialNB()
model_nb.fit(X_train, y_train)

# Train set scores
scores_train = model_nb.score(X_train, y_train)
print("MultinomialNB train set accuracy: ", scores_train)  # 94.15%

# Val set scores
scores_val = model_nb.score(X_val, y_val)
print("MultinomialNB val set accuracy: ", scores_val)  # 80.39%


## Logistic Regression
print("\nLogistic Regression")
model_lr  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_lr.fit(X_train, y_train)

# Train set scores
scores_train = model_lr.score(X_train, y_train)
print("Train set accuracy: ", scores_train)  # 98.05%

# Val set scores
scores_val = model_lr.score(X_val, y_val)
print("Val set accuracy: ", scores_val)  # 78.78%

# Using Logistic Regression, the results improved in the training accuracy from 94.15%
# to 98.05%, however, in the validation accuracy was reduced by 1.61%


## Add bigram features to the Logistic Regression

# Changing Vectorization
# Train
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer2 = CountVectorizer(ngram_range=(1, 2))    # ngram_range = (1,2) will include unigram and bigram
X_train = vectorizer2.fit_transform(train)
X_train = X_train.toarray()

# Val
val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer2.transform(val)
X_val = X_val.toarray()

# Model, one bigrams are added
model_lr_bgm  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_lr_bgm.fit(X_train, y_train)

# Train set scores
scores_train = model_lr_bgm.score(X_train, y_train)
print("Including Bigrams train set accuracy: ", scores_train)  # 99.97%

# Val set scores
scores_val = model_lr_bgm.score(X_val, y_val)
print("Including Bigrams val set accuracy: ", scores_val)  # 79.24%

# When we add bigrams the accuracy of both sets increase, this makes sense because
# now we are using unigrams and also bigrams, so our results are more robust.


### Modifying features

## 1) Removing stop words, including unigrams and bigrams
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer3 = CountVectorizer(stop_words='english', ngram_range=(1, 2))    # include unigram and bigram
X_train = vectorizer3.fit_transform(train)
X_train = X_train.toarray()

val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer3.transform(val)
X_val = X_val.toarray()

model_no_stopwords  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_no_stopwords.fit(X_train, y_train)

# Train set scores
scores_train = model_no_stopwords.score(X_train, y_train)
print("No stop words train set accuracy: ", scores_train)  # 99.52%

# Val set scores
scores_val = model_no_stopwords.score(X_val, y_val) 
print("No stop words val set accuracy: ", scores_val)  # 76.37%

## 2) Binaring the feature counts, including unigramas, bigrams
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer4 = CountVectorizer(binary=True, ngram_range=(1, 2))    # include unigram and bigram
X_train = vectorizer4.fit_transform(train)
X_train = X_train.toarray()

val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer4.transform(val)
X_val = X_val.toarray()

model_binary  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_binary.fit(X_train, y_train)

# Train set scores
scores_train = model_binary.score(X_train, y_train)
print("Binary NB train set accuracy: ", scores_train)  # 99.97%

# Val set scores
scores_val = model_binary.score(X_val, y_val) 
print("Binary NB val set accuracy: ", scores_val)  # 78.89%


## 3) Including trigrams, binary NB
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer5 = CountVectorizer(binary=True, ngram_range=(1, 3))    # include 1-gram,2-gram,3-gram
X_train = vectorizer5.fit_transform(train)
X_train = X_train.toarray()

val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer5.transform(val)
X_val = X_val.toarray()

model_trigram  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_trigram.fit(X_train, y_train)

# Train set scores
scores_train = model_trigram.score(X_train, y_train)
print("Trigram, Binary NB train set accuracy: ", scores_train)  # 100%

# Val set scores
scores_val = model_trigram.score(X_val, y_val) 
print("Trigram, Binary NB val set accuracy: ", scores_val)  # 79.47%


## 4) TF-IDF, trigram
train = []
for seq in train_sents:
    train.append(' '.join(seq))
vectorizer6 = CountVectorizer(ngram_range=(1, 3))    # include 1-gram,2-gram,3-gram
X_train = vectorizer6.fit_transform(train)
X_train = X_train.toarray()

val = []
for seq in val_sents:
    val.append(' '.join(seq))
X_val = vectorizer6.transform(val)
X_val = X_val.toarray()

# From ocurrences to frequencies
# Train set 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

# Val set
X_val_tfidf = tfidf_transformer.transform(X_val)

model_tfidf  = LogisticRegression(solver='liblinear')    # solver='liblinear'
model_tfidf.fit(X_train, y_train)

# Train set scores
scores_train = model_tfidf.score(X_train, y_train)
print("TF-IDF, Trigram train set accuracy: ", scores_train)  # 100%

# Val set scores
scores_val = model_tfidf.score(X_val, y_val) 
print("TF-IDF, Trigram val set accuracy: ", scores_val)  # 79.70%

'''
Results summary
Base line train set accuracy:  0.5216763005780347
Base line val set accuracy:  0.5091743119266054
MultinomialNB train set accuracy:  0.9414739884393064
MultinomialNB val set accuracy:  0.8038990825688074

Logistic Regression
Train set accuracy:  0.9804913294797688
Val set accuracy:  0.7878440366972477
Including Bigrams train set accuracy:  0.9997109826589595
Including Bigrams val set accuracy:  0.7924311926605505
No stop words train set accuracy:  0.9952312138728324
No stop words val set accuracy:  0.7637614678899083
Binary NB train set accuracy:  0.9997109826589595
Binary NB val set accuracy:  0.7889908256880734
Trigram, Binary NB train set accuracy:  1.0
Trigram, Binary NB val set accuracy:  0.7947247706422018
TF-IDF, Trigram train set accuracy:  1.0
TF-IDF, Trigram val set accuracy:  0.7970183486238532
'''