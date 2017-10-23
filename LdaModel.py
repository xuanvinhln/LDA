"""
Model of LDA
"""
import numpy as np
import numpy.random as rnd
from scipy import log

class LdaSufficientStats(object):
    """
    LDA_ss
    """
    def __init__(self, n_topics, size_vocab):
        self.alpha_ss = 0.0
        self.class_words = np.empty((n_topics, size_vocab))
        self.class_total = np.empty(n_topics)
        self.num_docs = 0

    def random_initialize_ss(self, n_topics, size_vocab):
        """
        Random_init_ss
        """
        for k in xrange(n_topics):
            for n in xrange(size_vocab):
                self.class_words[k, n] = 1.0/size_vocab + rnd.random()
        self.class_total = self.class_words.sum(axis=1)

    def zero_initialize_ss(self):
        """
        zero_init_ss
        """
        self.alpha_ss = 0.0
        self.class_words.fill(0.0)
        self.class_total.fill(0.0)
        self.num_docs = 0

class LdaModel(object):
    """
    LDA model class
    """
    def __init__(self, n_topics=0, size_vocab=0):
        self.alpha = 1.0
        self.log_prob_w = np.empty((n_topics, size_vocab))
        self.num_topics = n_topics
        self.size_vocab = size_vocab

    def maximum_likelihood(self, ss):
        """
        Update Beta
        """
        # cap nhat Beta
        for k in xrange(self.num_topics):
            for w in xrange(self.size_vocab):
                if (ss.class_words[k, w] > 0):
                    self.log_prob_w[k, w] = log(ss.class_words[k, w])\
                                           - log(ss.class_total[k])
                else:
                    self.log_prob_w[k, w] = -100

    def save(self, filename):
        """
        Save LDA model
        """
        f_name = filename+".beta"
        filestream = open(f_name, 'w')
        for i in xrange(self.num_topics):
            filestream.write(" ".join("{0:5.10f}".format(b) for b in self.log_prob_w[i]))
            filestream.write("\n")
        filestream.close()

        f_name = filename+".other"
        filestream = open(f_name, 'w')
        filestream.write("num topics: {0}\n".format(self.num_topics))
        filestream.write("vocab size: {0}\n".format(self.size_vocab))
        filestream.write("alpha: {0:5.10f}\n".format(self.alpha))
        filestream.close()

    def load(self, filename):
        """
        Load model from file
        """
        print "Load model"

        f_name = filename+".other"
        print "loading "+f_name
        filestream = open(f_name, 'r')
        n_topics = int(filestream.readline())
        size_vocab = int(filestream.readline())
        alpha = float(filestream.readline())
        print (n_topics, size_vocab, alpha)
        filestream.close()

        self.alpha = alpha
        self.num_topics = n_topics
        self.size_vocab = size_vocab

        f_name = filename+".beta"
        print "loading "+f_name
        self.log_prob_w = np.genfromtxt(f_name, delimiter=" ")
        