import sqlite3
import h5py
import chess


def pgn_db_to_hdf5(pgn_db, hdf5_fp):
    """Creates an hdf5 file with 2 databases: x, and y.
    x is an array of all positions ever reached.
    y is an array of the number of games that included that position, and how many of them white won.

    For example, if x[0] is a position, then y[0, 0] is how many times that position has been played in a game, and y[0, 1] is how
    many of those games have been won.

    The x database has shape: (n, 384)
    The y database has shape: (n, 2)"""

    # TODO: All of it
    pass
