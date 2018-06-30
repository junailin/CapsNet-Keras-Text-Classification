import os
import argparse
import numpy as np
from keras import callbacks
from keras.utils.vis_utils import plot_model
from data_helpers import load_imdb
from utils import plot_log
from keras import backend as K
from capsule_net import CapsNet

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

def train(model, data, args):
    X, Y = data

    # Callbacks
    log = callbacks.CSVLogger(filename=args.save_dir + '/log.csv')

    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size,
                               histogram_freq=args.debug)

    checkpoint = callbacks.ModelCheckpoint(filepath=args.save_dir + '/weights-improvement-{epoch:02d}.hdf5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss],
                  metrics=['accuracy'])

    model.fit(x=X,
              y=Y,
              validation_split=0.2,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=[log, tb, checkpoint, lr_decay],
              shuffle=True,
              verbose=1)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)
    return model


def evaluation(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)

    print('-' * 50)

    # model.compile(optimizer='adam',
    #               loss=[margin_loss, 'mse'],
    #               loss_weights=[1., args.lam_recon],
    #               metrics={'out_caps': 'accuracy'})
    #
    # score = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # print('Test acc:', np.sum( np.argmax(y_pred) == np.argmax(y_test) ) / y_test.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lam_recon', default=0.0005, type=float)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_imdb()

    print(x_train.shape)
    print(y_train.shape)

    # define model
    model = CapsNet(input_shape=x_train.shape[1:], n_class=2, num_routing=args.num_routing)

    model.summary()
    plot_model(model, to_file=args.save_dir + '/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)

    if args.is_training:
        train(model=model, data=(x_train, y_train), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        evaluation(model=model, data=(x_test, y_test))