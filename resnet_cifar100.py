"""Train ConvNet Mean Teacher on CIFAR-10 training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/cifar10_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""

import logging
import tensorflow as tf
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_float('entropy_reg', 0.01, 'Entropy regularization coefficient.')

from datetime import datetime

from run_context import RunContext
from datasets import Cifar100ZCA
#from mean_teacher.model import Model
from myresnet_model import Model
import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 'all' #10000

    print('Running')

    model = Model(RunContext(__file__, 0))
    model['flip_horizontally'] = True
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 0
    model['rampup_length'] = 5000
    model['training_length'] = 400000
    model['max_consistency_cost'] = 3000.0
    model['entropy_factor'] = FLAGS.entropy_reg

    print('Reg:' , FLAGS.entropy_reg)
    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar100ZCA(data_seed, n_labeled)
    training_batches = minibatching.training_batches(cifar.training, n_labeled_per_batch=50)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
