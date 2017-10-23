"""
Main program
"""
import sys
import time
from LdaEstimator import run_EM
from Corpus import Corpus

def main():
    """
    main function
    """
    start_time = time.time()
    if len(sys.argv) != 6:
        print "usage: python main.py <init_alpha> <modeldir_name> <num_topic> <data_file> <random/load>"
        sys.exit(1)

    init_alpha = float(sys.argv[1])
    directory = sys.argv[2]
    num_topics = int(sys.argv[3])
    data_file = sys.argv[4]
    start_type = sys.argv[5]

    # read_data
    corpus = Corpus()
    corpus.read_data(data_file)

    # Run LDA
    run_EM(init_alpha, directory, num_topics, corpus, start_type)
    # run_EM(directory, corpus, start_type)
    end_time = time.time()
    end_time = (end_time - start_time)/60
    print "Time run of program {:f} minutes".format(end_time)
    filename = directory+"/Time_Running.time"
    time_file = open(filename, 'w')
    time_file.write("Time run of program {:f} minutes".format(end_time))
    time_file.close()

if __name__ == "__main__":
    main()
    