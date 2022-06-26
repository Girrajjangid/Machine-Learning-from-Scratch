import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# To clear the session
def cls():
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()

# Loading the dataset
(train_set_raw,test_set_raw,valid_set_raw),info  = tfds.load("tf_flowers",split=["train[:80%]","train[80%:90%]","train[90%:]"],as_supervised=True,with_info=True)
