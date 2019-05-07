# -*- coding: utf-8 -*-


def print_config(config):
    """Print config used in this model.
    """
    print("=> Config Settings <=")
    print("Learning rate:", config.learning_rate)
    print("L2 normalization:", config.l2_norm)
    print("Batch size:", config.batch_size)
    print("Embedding size:", config.k)
    print("Total epoch:", config.num_epoch)
    print("=> Config Settings <=")
