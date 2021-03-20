import chess
import chess.pgn
import sqlite3
import h5py


def pgn_db_to_hdf5(pgn_fp, hdf5_fp):
    """Creates an hdf5 file with 2 databases: x, and y.
    x is an array of all positions ever reached.
    y is an array of the number of games that included that position, and how many of them white won.
    For example, if x[0] is a position, then y[0, 0] is how many times that position has been played in a game,
    and y[0, 1] is how many of those games have been won.

    The x database has shape: (n, 903); 8 * 8 * (6 + 6 + 2) + 7.
    The y database has shape: (n, 2)"""

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
