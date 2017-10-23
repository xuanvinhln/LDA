"""
Corpus and document class
"""
import os
import re

class Document(object):
    """
    Document class
    """
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

class Corpus(object):
    """
    Corpus class
    """
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = 0

    def read_data(self, filename):
        """
        read data from file
        """
        if not os.path.exists(filename):
            print 'no data file, please check it'
            return
        print 'reading data from %s.' % filename

        for line in file(filename):
            ss = line.strip().split()
            if len(ss) == 0:
                continue

            doc = Document()
            doc.length = int(ss[0])
            doc.words = [0 for w in range(doc.length)]
            doc.counts = [0 for w in range(doc.length)]
            for w, pair in enumerate(re.finditer(r"(\d+):(\d+)", line)):
                doc.words[w] = int(pair.group(1))
                doc.counts[w] = int(pair.group(2))

            doc.total = sum(doc.counts)
            self.docs.append(doc)

            if doc.length > 0:
                max_word = max(doc.words)
                if max_word >= self.size_vocab:
                    self.size_vocab = max_word + 1

            # if len(self.docs) >= 10000:
            #     break

        self.num_docs = len(self.docs)
        print "finished reading %d docs." % self.num_docs
        print "num doc of corpus %d " % self.num_docs
        print "size_vocab %d " % self.size_vocab
        