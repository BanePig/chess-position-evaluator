import argparse
import math
import sys

import chess

import data
import model
import os
import tensorflow as tf
import h5py
import numpy as np
import eval
import chess.pgn
import random


class DataSeq(tf.keras.utils.Sequence):
    def __init__(self, fp, batch_size):
        self.h5_f = h5py.File(fp, 'r')
        self.x_board_db = self.h5_f['x_board']
        self.x_misc_db = self.h5_f['x_misc']
        self.y_db = self.h5_f['y']
        self.batch_size = batch_size
        self.indices = list(range(len(self)))
        random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(self.x_board_db.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        sidx = self.indices[idx]

        x_board = self.x_board_db[sidx:sidx + self.batch_size]
        x_misc = self.x_misc_db[sidx:sidx + self.batch_size]
        y = self.y_db[sidx:sidx + self.batch_size]
        x = np.concatenate([x_board, x_misc], axis=1)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.indices)


if __name__ == '__main__':
    sys.argv.pop(0)

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data', nargs=1, required=True, dest='train_data')
    parser.add_argument('-test_data', nargs=1, required=False, dest='test_data')
    parser.add_argument('-o', nargs=1, required=True, dest='output_file')
    parser.add_argument('-w', nargs=1, required=False, dest='initial_weights', type=str)
    parser.add_argument('-epochs', nargs=1, required=False, default=1, dest='epochs', type=int)
    parser.add_argument('--ignore_no_weights', required=False, default=False, dest='ignore_no_weights', const=True, action='store_const')

    cli_args = parser.parse_args(sys.argv)

    network = model.create_model()

    batch_num = 0

    def save_and_eval(*args, **kwargs):
        global batch_num
        network.reset_metrics()
        network.save_weights(cli_args.output_file[0])
        print("Weights saved.")
        with open('./test.pgn', 'r') as f:
            game = chess.pgn.read_game(f)
        headers = game.headers
        while game is not None:
            net_in = data.get_full_features(game, game.board(), headers)
            net_in = net_in.reshape(1, *net_in.shape)
            out = network(net_in).numpy()
            if game.move is not None:
                print(game.move.uci() + ' ', end='')
            print(f'Eval: {out}')
            game = game.next()


    train_seq = DataSeq(cli_args.train_data[0], 1024)
    if cli_args.test_data is not None and len(cli_args.test_data) > 0:
        test_seq = DataSeq(cli_args.test_data[0], 1024)
    else:
        test_seq = None

    if cli_args.initial_weights is not None:
        if cli_args.ignore_no_weights and not os.path.exists(cli_args.initial_weights[0]):
            print("No weights file. Ignoring...")
        else:
            network(train_seq[0][0])
            network.load_weights(cli_args.initial_weights[0])

    network.fit(train_seq, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=save_and_eval)
    ], validation_data=test_seq, epochs=cli_args.epochs[0])
