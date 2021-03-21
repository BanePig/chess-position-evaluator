import chess
import chess.pgn
import sqlite3
import io
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import time
from multiprocessing import Pool, Process


def read_large_pgn(fp, ignore_abandoned):
    with open(fp, "r") as pgn_file:
        pgn = ""
        num_blank_lines = 0
        invalid = False
        for line in pgn_file:
            if line == "[Termination \"Abandoned\"]\n":
                invalid = True
                continue
            if line.strip() == "":
                num_blank_lines += 1
            if num_blank_lines == 2:
                if not invalid or not ignore_abandoned:
                    yield pgn
                invalid = False
                pgn = ""
                num_blank_lines = 0
                continue
            pgn += line


def get_board_features(board):
    board_arr = np.zeros(shape=(15, 8, 8), dtype=np.uint8)

    if board.turn == chess.BLACK:
        board_aligned = board.mirror()
    else:
        board_aligned = board

    bitboard_idx = 0
    # Set player one pieces
    for player in (chess.WHITE, chess.BLACK):
        for piece_id in range(1, 7):
            for square in board_aligned.pieces(piece_id, player):
                board_arr[bitboard_idx, square // 8, square % 8] = 1
            bitboard_idx += 1

    # Set repetitions
    # Temporarily removed to improve performance.
    # for move in board.pseudo_legal_moves:
    #     board.push(move)
    #     if board.can_claim_threefold_repetition():
    #         board_arr[bitboard_idx, move.from_square // 8, move.from_square % 8] = 1
    #         board_arr[bitboard_idx, move.to_square // 8, move.to_square % 8] = 1
    #     else:
    #         for move2 in board.pseudo_legal_moves:
    #             board.push(move2)
    #             if board.can_claim_threefold_repetition():
    #                 board_arr[bitboard_idx + 1, move.from_square // 8, move.from_square % 8] = 1
    #                 board_arr[bitboard_idx + 1, move.to_square // 8, move.to_square % 8] = 1
    #             board.pop()
    #     board.pop()
    # bitboard_idx += 2

    # Set en-passante mask
    for move in board_aligned.pseudo_legal_moves:
        if board_aligned.is_en_passant(move):
            square = move.to_square
            board_arr[bitboard_idx, square // 8, square % 8] = 1

    return np.reshape(board_arr, newshape=(960,))


def get_misc_features(game, board, headers):
    x_misc = np.zeros(shape=(10,))
    x_misc[0] = int(headers.get('WhiteElo' if board.turn else 'BlackElo') or -100) / 3000
    x_misc[1] = int(headers.get('BlackElo' if board.turn else 'WhiteElo') or -100) / 3000
    x_misc[2] = (min(game.clock() or 150 * 60, 150 * 60)) / 600
    if game.next() is not None and game.next().clock() is not None:
        x_misc[3] = (min(game.next().clock(), 150 * 60)) / 600
    else:
        x_misc[3] = 15
    x_misc[4] = board.has_kingside_castling_rights(board.turn)
    x_misc[5] = board.has_queenside_castling_rights(board.turn)
    x_misc[6] = board.has_kingside_castling_rights(not board.turn)
    x_misc[7] = board.has_queenside_castling_rights(not board.turn)
    x_misc[8] = board.halfmove_clock / 50
    try:
        x_misc[9] = int(headers['TimeControl'].split('+')[1]) / 5
    except IndexError:
        x_misc[9] = 0
    return x_misc


def get_full_features(game, board, headers):
    board_features = get_board_features(board)
    misc_features = get_misc_features(game, board, headers)
    return np.concatenate([board_features, misc_features], axis=0)


def add_to_db(processed_data, h5_f, x_board_db, x_misc_db, y_db):
    for data in processed_data:
        size = x_board_db.attrs['size']
        if size >= x_board_db.shape[0]:
            x_board_db.resize((x_board_db.shape[0] + 16384, x_board_db.shape[1]))
            x_misc_db.resize((x_board_db.shape[0], x_misc_db.shape[1]))
            y_db.resize((x_board_db.shape[0], y_db.shape[1]))
            h5_f.flush()
            print(f"Total positions processed={size}")
        x_board_db[size] = data[0]
        x_misc_db[size] = data[1]
        y_db[size] = data[2]
        x_board_db.attrs['size'] += 1


def process(pgn):
    out = []
    game = chess.pgn.read_game(io.StringIO(pgn))
    headers = game.headers
    winner = None
    pgn_stripped = pgn.strip('\n ')
    if pgn_stripped.endswith('1-0'):
        winner = chess.WHITE
    elif pgn_stripped.endswith('0-1'):
        winner = chess.BLACK
    while game is not None:
        board = game.board()

        x_board = get_board_features(board)

        x_misc = get_misc_features(game, board, headers)

        y = np.zeros(shape=(3,))
        if winner == board.turn:
            y[0] = 1
        elif winner is None:
            y[1] = 1
        else:
            y[2] = 1
        out.append((np.reshape(x_board, newshape=(960,)), x_misc, y))
        game = game.next()
    return out


def append_pgn_to_hdf5(pgn_fp, hdf5_fp, ignore_non_unique):
    """Creates or appends to an hdf5 file with 3 databases: x_board, x_misc, and y.
    x_board is an array of board positions, representing piece positions and repetitions.
    x_misc is an array of numbers, which are potentially useful features to the AI, such as ELO and clock time.
    y is a one-hot encoded array of whether or not the game ended in a win, loss, or draw.
    For example, if x_board[0] is a position (encoding), then y[0, 0] is whether it was won, and
    y[0, 1] is whether it was lost.

    Note that the hdf5 cannot have an x and y database already, unless it was previously created by this function.

    The x_board database has shape: (n, 960).
    The x_misc database has shape: (n, 9).
    The y database has shape: (n, 3)"""

    if os.path.exists(hdf5_fp):
        h5_f = h5py.File(hdf5_fp, 'r+')
        if 'x_board' not in h5_f or 'size' not in h5_f['x_board'].attrs:
            print("ERROR: h5_f is an invalid database! Please use another file path or delete the file.")
    else:
        h5_f = h5py.File(hdf5_fp, 'w')
        print("HDF5 file does not exist, creating one...")
        d = h5_f.create_dataset('x_board', shape=(0, 960), chunks=(16384, 960), maxshape=(None, 960), dtype=np.bool)
        d.attrs['size'] = 0
        d.attrs['included_files'] = ''
        h5_f.create_dataset('x_misc', shape=(0, 10), chunks=(16384, 10), maxshape=(None, 10), dtype=np.single)
        h5_f.create_dataset('y', shape=(0, 3), chunks=(16384, 3), maxshape=(None, 3), dtype=np.ushort)
        del d

    x_board_db = h5_f['x_board']
    x_misc_db = h5_f['x_misc']
    y_db = h5_f['y']

    if not ignore_non_unique and pgn_fp in x_board_db.attrs['included_files'].split('\n'):
        raise ValueError(f"The file {pgn_fp} has already been added to this hdf5 database. "
                         f"Set ignore_non_unique to true to ignore.")
    else:
        x_board_db.attrs['included_files'] += pgn_fp + '\n'

    with Pool(15) as pool:
        cache = []
        for next_pgn in read_large_pgn(pgn_fp, ignore_abandoned=True):
            cache.append(next_pgn)
            if len(cache) >= 1024:
                out = [item for sublist in pool.map(process, cache) for item in sublist]
                cache = []
                add_to_db(out, h5_f, x_board_db, x_misc_db, y_db)


if __name__ == '__main__':
    sys.argv.pop(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs=1, required=True, dest='input', type=str)
    parser.add_argument('-o', nargs=1, required=True, dest='output', type=str)
    parser.add_argument('--ignore_non_unique', action='store_const', dest='ignore_non_unique', const=True,
                        default=False)

    args = parser.parse_args(sys.argv)
    if os.path.isdir(args.input[0]):
        for fp in os.listdir(args.input[0]):
            append_pgn_to_hdf5(os.path.join(args.input[0], fp), args.output[0],
                               ignore_non_unique=args.ignore_non_unique)
