#!/opt/anaconda3/bin/python
#import  sys
#sys.path.append('/Users/randallhelzerman/projects/src/matrix_net/bdol-ml/utils')

import pickle, gzip
from data_utils import create_minibatches
from neural_network import MLP

f = gzip.open('/Users/randallhelzerman/Downloads/archive//mnist.pkl.gz')
train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
f.close()

minibatch_size = 100
print("Creating data...")
train_data, train_labels = create_minibatches(train_set[0], train_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)
valid_data, valid_labels = create_minibatches(valid_set[0], valid_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)
print("Done!")


mlp = MLP(layer_config=[784, 100, 100, 10], minibatch_size=minibatch_size)
mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
             eval_train=True)
