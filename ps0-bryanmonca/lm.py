from nltk import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.util import flatten
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
import matplotlib.pyplot as plt
from math import log2


## Importing data and tokenizing it
# Train data
file = open('/Users/bryanmonca/Workspace/cmsi-537/brown-train.txt', 'rt')
train_corpus = file.read()
file.close()

# tokenize into sentences
sentences = sent_tokenize(train_corpus) 
tokenized_text = []
for seq in sentences:
    tokenized_text.append(list(map(str.lower, word_tokenize(seq))))

# add start and end of each sentence
for seq in tokenized_text:
    seq.insert(0,'<s>')
    seq.append('</s>')

# Val data
file = open('/Users/bryanmonca/Workspace/cmsi-537/brown-val.txt', 'rt')
val_corpus = file.read()
file.close()

# tokenize into sentences
val_sentences = sent_tokenize(val_corpus) 
val_tokenized_text = []
for seq in val_sentences:
    val_tokenized_text.append(list(map(str.lower, word_tokenize(seq))))


# add start and end of each sentence
for seq in val_tokenized_text:
    seq.insert(0,'<s>')
    seq.append('</s>')


## Plotting frequencies of each word
# tokenize into words
word_tokens = word_tokenize(train_corpus)  # To be plotted

# Frequency Distribution Plot
fdist = FreqDist(word_tokens)
fdist.plot(40,cumulative=False) # top 40 words
plt.show()

'''
After making a plot of the words in our vocabulary vs. the frequency of each 
word, we can see a long tail pattern. The frequency of the words decreases. 
The words most commonly used in english are the ones that have more frequency in 
our training data. The only exception is the token UNK, which have the most 
common cases. This distribution follows Zipf's law. 
'''


## Counting bigrams and context of them
bigram_counts = {}
context_counts = {}
for seq in tokenized_text:
    for i in range(1, len(seq)):
        bigram = seq[i-1] + " " + seq[i]
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1
            
        context = seq[i-1]
        if context in context_counts:
            context_counts[context] += 1
        else:
            context_counts[context] = 1  

## Perplexities calculations using bigram probabilities with add-α smoothing
def perplexity(tokenized_data, bigram_prob, context_counts, alpha):
    acum = 0
    for sent in tokenized_data:
        for i in range(1, len(sent)):
            bigram = sent[i-1] + " " + sent[i]
            if bigram not in bigram_prob:
                context = sent[i-1]
                if context in context_counts:
                    acum += log2(alpha / (context_counts[context] + (alpha * len(context_counts))))
            elif bigram in bigram_prob:
                acum += log2(bigram_prob[bigram])
    return 2 ** ((-1/len(flatten(tokenized_data))) * acum)


## Training using add-α smoothing for different α's 
## Then, calculating perplexities for train and val set
alphas = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10]
train_ppl = []
val_ppl = []
for alpha in alphas:
    bigram_prob = {}
    for bigram, counts in bigram_counts.items():
        words = bigram.split(" ")
        context = words[0]
        bigram_prob[bigram] = (bigram_counts[bigram] + alpha) / (context_counts[context] + (alpha*len(context_counts)))
    # Train set perplexities
    train_ppl.append(perplexity(tokenized_text, bigram_prob, context_counts, alpha))
    val_ppl.append(perplexity(val_tokenized_text, bigram_prob, context_counts, alpha))

## Print results
for i in range(len(alphas)):
    print("Train set ppl with alpha", alphas[i], "=", train_ppl[i])
print("")
for i in range(len(alphas)):
    print("Val set ppl with alpha", alphas[i], "=", val_ppl[i])

## Plotting train, val perplexities vs alpha
fig = plt.figure()
plt.plot(train_ppl, alphas, label='train')
plt.plot(val_ppl, alphas, label='val')
plt.legend(loc='upper left')
plt.show()

# Lower alphas get better perplexities for the train and val set.
# The best model has alphas less than 1.
# Best PPL result for train set is with alpha=10e-5; for the val set
# is 10e-3


## Adjusting Training data and fixing alpha, and obtaining perplexities
train_ppl = []
val_ppl = []
alpha = 0.001
for j in range(1, 11):
    ## Counting bigrams and context of them
    bigram_counts = {}
    context_counts = {}
    fraction = int((j/10) * len(tokenized_text))    
    for seq in tokenized_text[:fraction]: 
        for i in range(1, len(seq)):
            bigram = seq[i-1] + " " + seq[i]
            if bigram in bigram_counts:
                bigram_counts[bigram] += 1
            else:
                bigram_counts[bigram] = 1

            context = seq[i-1]
            if context in context_counts:
                context_counts[context] += 1
            else:
                context_counts[context] = 1 

    ## Training using add-α smoothing for different α's 
    ## Then, calculating perplexities for train and val set 
    bigram_prob = {}
    for bigram, counts in bigram_counts.items():
        words = bigram.split(" ")
        context = words[0]
        bigram_prob[bigram] = (bigram_counts[bigram] + alpha) / (context_counts[context] + (alpha*len(context_counts)))
    # Train set perplexities
    train_ppl.append(perplexity(tokenized_text, bigram_prob, context_counts, alpha))
    val_ppl.append(perplexity(val_tokenized_text, bigram_prob, context_counts, alpha))

## Print results
print("\n")
fractions = [str(i*10)+"%"  for i in range(1,11)]
for i in range(len(fractions)):
    print("Train set ppl with alpha=0.001,", fractions[i], "train data:", train_ppl[i])
print("")
for i in range(len(fractions)):
    print("Val set ppl with alpha=0.001,", fractions[i], "train data:", val_ppl[i])

## Plotting train, val perplexities vs fraction percentage
fig = plt.figure()
plt.plot(train_ppl, fractions, label='train')
plt.plot(val_ppl, fractions, label='val')
plt.legend(loc='upper left')
plt.show()

# Less training data gets a lower perplexity value for both sets
# The more data we have, the better we can predict the next word
# in a sentence



### Sherlock Holmes - Generation ###
# Data available in "ps0-bryanmonca/lm-data/s_holmes.txt"
file = open('/Users/bryanmonca/Files/LMU/CMSI_537_FALL/Assignments/ps0-bryanmonca/lm-data/s_holmes.txt', 'rt')
train_corpus = file.read()
file.close()

# Tokenizing
tokenized_text = []
for seq in sent_tokenize(train_corpus):
    tokenized_text.append(list(map(str.lower, word_tokenize(seq))))

## Bigram Model
# Preprocessing
n = 2 # bigrams
# pad sequences and preprocess train data and vocabulary generators  
# with unigrams and bigrams
train_data, padded_seq = padded_everygram_pipeline(n, tokenized_text)  

# Model: Maximum Likelihood Estimator (MLE)
model = MLE(n)  # bigram model

# Training
model.fit(train_data, padded_seq)

# Extra Step - to convert generated tokens into a sentence
detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed):
    text = []
    for token in model.generate(num_words, random_seed=random_seed):
        if (token == '<s>'):
            continue
        if (token == '</s>'):
            break
        text.append(token)
    print(detokenize(text))

# Generation
print("\nGeneration based on bigrams\n")

generate_sent(model, num_words=50, random_seed=15)
print("\n")
'''
“ absolutely quiet? ” asked sherlock holmes the ‘ his fangs upon it expressly 
for three thousand will bring you in five little knot in some illness through 
which present case there ’ s wharf, and down upon the red and most solemn 
mr. fowler and lived rent
'''

generate_sent(model, num_words=40, random_seed=65)
print("\n")
'''
he gasped the lamp from prague for standing smiling, no change my notes, 
and now that they were at least interested in finding what of managing it 
struck me the general shriek of his conclusions before he was
'''

generate_sent(model, num_words=45, random_seed=10)
print("\n")
'''
moment her maid, where they seem to me a few years ’ clock ticking loudly 
expressed a question in the passage-lamp your room while wooden-leg had a 
baboon, ” said he was done no doubt, one to boscombe pool.
'''

## Trigram LM
print("\nGeneration based on trigrams\n")
n = 3
train_data, padded_seq = padded_everygram_pipeline(n, tokenized_text)  
model = MLE(n)  # trigram model
model.fit(train_data, padded_seq)

generate_sent(model, num_words=50, random_seed=10)
print("\n")
'''
like fear sprang up in the office, my dear fellow wanted to do with the ice 
crystals.
'''

generate_sent(model, num_words=50, random_seed=20)
print("\n")
'''
what purpose that could answer i confess that i could not wonder at 
lestrade ’ s better! ” holmes moved the lamp away from the body.
'''

generate_sent(model, num_words=50, random_seed=40)
print("\n")
'''
he was, by winding up every meal by taking out the case into your hands 
now—so far as he was a movement and an appropriate dress, i slipped in in 
safety.
'''
generate_sent(model, num_words=64, random_seed=52)
print("\n")
'''
“ and what have you her photograph, your lad tugging at one door of a brave 
fellow, i suppose that you will remember that not only to put colour and life 
into each of which were submitted to him, however, in case i found her more 
interesting than her little problem which you were when you go to the court-yard
'''

# Comments on Text Generation

# Using trigrams make the generation closer to the style of the book, with
# better sentences, this is super interesting, only using trigrams we
# could create fake Sherlock Holmes stories/phrases.
