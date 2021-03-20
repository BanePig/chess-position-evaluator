import chess
import chess.pgn
import sqlite3
import h5py
import numpy as np
import os


def append_pgn_to_hdf5(pgn_fp, hdf5_fp):
    """Creates or appends to an hdf5 file with 3 databases: x_board, x_misc, and y.
    x_board is an array of board positions, representing piece positions and repititions.
    x_misc is an array of numbers, which are potentially useful features to the AI, such as ELO and clock time.
    y a one-hot encoded array of whether or not the game ended in a win, loss, or draw.
    For example, if x[0] is a position (encoding), then y[0, 0] whether it was won, and y[0, 1] whether it was lost.

    Note that the hdf5 cannot have an x and y database already, unless it was previously created by this function.

    The x_board database has shape: (n, 896).
    The x_misc database has shape: (n, 7).
    The y database has shape: (n, 3)"""

    if os.path.exists(hdf5_fp):
        h5_f = h5py.File(hdf5_fp, 'r+')
    else:
        h5_f = h5py.File(hdf5_fp, 'w')

    if 'x_board' not in h5_f:
        h5_f.create_dataset('x_board', shape=(1000, 896), maxshape=(None, 896), dtype=np.bool)
    if 'x_misc' not in h5_f:
        h5_f.create_dataset('x_misc', shape=(1000, 7), maxshape=(None, 7), dtype=np.single)
    if 'y' not in h5_f:
        h5_f.create_dataset('y', shape=(1000, 3), maxshape=(None, 3), dtype=np.ushort)

    with open(pgn_fp, "r") as pgn_file:
        pgn = ""
        num_blank_lines = 0
        index = 0
        invalid = False
        for line in pgn_file:
            if line == "[Termination \"Abandoned\"]\n":
                invalid = True
                continue
            if index > 10:
                break
            if line.strip() == "":
                num_blank_lines += 1
            if num_blank_lines == 2:
                if not invalid:
                    print(pgn)
                invalid = False
                index += 1
                pgn = ""  # Keep track of this somehow
                num_blank_lines = 0
                continue
            pgn += line


if __name__ == '__main__':
    # Test
    append_pgn_to_hdf5('./data/lichess_db_standard_rated_2017-04.pgn', './db.h5')
