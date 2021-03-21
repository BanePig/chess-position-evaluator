import argparse
import sys
import chess
import chess.pgn
import io
import model
import data
import numpy as np

if __name__ == '__main__':
    sys.argv.pop(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('pgn', nargs=1)
    parser.add_argument('-w', nargs=1, required=True, dest='weights', type=str)
    cli_args = parser.parse_args(sys.argv)

    with open(cli_args.pgn[0], 'r') as f:
        game = chess.pgn.read_game(f)

    net = model.create_model()
    net_initialized = False

    headers = game.headers
    while game is not None:
        net_in = data.get_full_features(game, game.board(), headers)
        net_in = net_in.reshape(1, *net_in.shape)
        if not net_initialized:
            net(net_in)
            net.load_weights(cli_args.weights[0])
            net_initialized = True
        out = net(net_in).numpy()
        print(game.board())
        print(f'Eval: {str(out)}')
        game = game.next()
