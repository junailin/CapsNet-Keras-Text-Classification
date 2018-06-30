from utils import plot_log
from matplotlib import pyplot
from keras import layers, models
from keras import backend as K
from capsule_layer import CategoryCap, PrimaryCap, Length, Mask

MAX_FEATURES = 5000
MAX_LEN = 400
EMBED_DIM = 50

def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)
    embed = layers.Embedding(MAX_FEATURES, EMBED_DIM, input_length=MAX_LEN)(x)

    conv1 = layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(embed)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primary_caps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")

    # Layer 3: Capsule layer. Routing algorithm works here.
    category_caps = CategoryCap(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='category_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(category_caps)

    return models.Model(input=x, output=out_caps)