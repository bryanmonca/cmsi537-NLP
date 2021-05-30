"""
Description: HMM with Viterbi decoding for named entity recognition.
Author: Dr. Korpusik
Reference: Chen & Narasimhan
Date: 6/29/2020
"""

import argparse
import numpy as np
import loader

from sklearn.metrics import f1_score


class HMM():
    """
    Hidden Markov Model (HMM) for named entity recognition.
    Two options for decoding: greedy or Viterbi search.
    """

    def __init__(self, dics, decode_type):
        self.num_words = len(dics['word_to_id'])
        self.num_tags = len(dics['tag_to_id'])

        # Initialize all start, emission, and transition probabilities to 1.
        self.initial_prob = np.zeros([self.num_tags])
        self.transition_prob = np.zeros([self.num_tags, self.num_tags])
        self.emission_prob = np.zeros([self.num_tags, self.num_words])
        self.decode_type = decode_type

    def train(self, corpus):
        """
        Trains a bigram HMM model using MLE estimates.
        Updates self.initial_prob, self.transition_prob, & self.emission_prob.

        The corpus is a list of dictionaries of the form:
        {'str_words': str_words,   # List of string words
        'words': words,            # List of word IDs
        'tags': tags}              # List of tag IDs
        The lists above all have the same length as that instance's sentence.
        """
        # Unigrams of tags
        tag_counts = np.zeros(self.num_tags)
        for sent in corpus:
            for tag in sent['tags']:
                tag_counts[tag] += 1 

        # Initial_prob, transition_prob, emission_prob
        for sent in corpus:
            tags, words = sent['tags'], sent['words']
            self.initial_prob[tags[0]] += 1
            self.emission_prob[tags[0], words[0]] += 1
            
            for i in range(1, len(tags)):
                self.transition_prob[tags[i-1], tags[i]] += 1
                self.emission_prob[tags[i], words[i]] += 1

        self.initial_prob = self.initial_prob / len(corpus)
        for tag in range(self.num_tags):
            self.transition_prob[tag] = self.transition_prob[tag] / tag_counts[tag]
            self.emission_prob[tag] = self.emission_prob[tag] / tag_counts[tag]

    def greedy_decode(self, sentence):
        """
        Decode a single sentence in greedy fashion.
        Return a list of tags.
        """
        tags = []

        start_scores = []
        for tag in range(self.num_tags):
            start_scores.append(self.initial_prob[tag] * 
                                self.emission_prob[tag][sentence[0]])
        tags.append(np.argmax(start_scores))

        for word in sentence[1:]:
            scores = []
            for tag in range(self.num_tags):
                scores.append(self.transition_prob[tags[-1]][tag] * 
                              self.emission_prob[tag][word])
            tags.append(np.argmax(scores))

        assert len(tags) == len(sentence)
        return tags

    def viterbi_decode(self, sentence):
        """
        Decode a single sentence using the Viterbi algorithm.
        Return a list of tags.
        Reference: http://www.cs.jhu.edu/~langmea/resources/lecture_notes/hidden_markov_models.pdf
        """
        tags = []
        mat = np.zeros([self.num_tags, len(sentence)])
        mat_bp = np.zeros([self.num_tags, len(sentence)], dtype=int)

        # First column of mat
        for tag in range(self.num_tags):
            mat[tag, 0] = self.initial_prob[tag] * self.emission_prob[tag, sentence[0]]

        # mat and mat_backpointer
        for j in range(1, len(sentence)):
            for i in range(self.num_tags):
                e_prob = self.emission_prob[i, sentence[j]]
                max_k, max_k_index = mat[0, j-1] * self.transition_prob[0, i] * e_prob, 0

                for i2 in range(1, self.num_tags):
                    prob = mat[i2, j-1] * self.transition_prob[i2, i] * e_prob
                    if prob > max_k:
                        max_k, max_k_index = prob, i2

                mat[i, j], mat_bp[i, j] = max_k, max_k_index

        # Find final max prob (starting point of backtrace)
        final_max_k, final_max_k_index = mat[0, len(sentence)-1], 0
        for i in range(1, self.num_tags):
            if mat[i, len(sentence)-1] > final_max_k:
                final_max_k, final_max_k_index = mat[i, len(sentence)-1], i

        # Backtrace
        tags.append(final_max_k_index)
        i = final_max_k_index
        for j in range(len(sentence)-1, 0, -1):
            i = mat_bp[i, j]
            tags.append(i)
        tags = tags[::-1]

        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence using a trained HMM.
        """
        if self.decode_type == 'viterbi':
            return self.viterbi_decode(sentence)
        else:
            return self.greedy_decode(sentence)


def evaluate(model, test_corpus, dics, args):
    """Predicts test data tags with the trained model, and prints accuracy."""
    num_correct = 0
    num_total = 0
    y_pred = []
    y_actual = []
    for i, sentence in enumerate(test_corpus):
        tags = model.tag(sentence['words'])

        str_tags = [dics['id_to_tag'][tag] for tag in tags]
        y_pred.extend(tags)
        y_actual.extend(sentence['tags'])

        num_correct += np.sum(np.array(tags) == np.array(sentence['tags']))
        num_total += len([w for w in sentence['words']])

    print('\nOverall accuracy:', (num_correct / num_total))
    return y_pred, y_actual


def main(args):
    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train', 
                                                lower=args.lower)

    # Train the HMM.
    model = HMM(dics, decode_type=args.decode_type)
    model.train(train_corpus)

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test', 
                                         lower=args.lower, word_to_id=dics['word_to_id'], 
                                         tag_to_id=dics['tag_to_id'])

    # Evaluate the model on the validation data.
    y_pred, y_actual = evaluate(model, test_corpus, dics, args)
    for tag, score in enumerate(f1_score(y_actual, y_pred, average=None)):
        print(dics['id_to_tag'][tag] + ' F1 score:', score)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--decode_type', default='greedy', choices=['viterbi', 'greedy'])

    args = parser.parse_args()
    main(args)