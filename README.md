# Chess Position Evaluator (BetaOne)

Hackathon project to evaluate a chess position using tensorflow, and a lot of data.

# Usage:

**YOU MUST USE PYTHON 3.6-3.8 TO INSTALL TENSORFLOW (AND THUS USE THE PROGRAM)**

- Install the requirements with pip install -r requirements.txt (From inside repository directory)
- Create a dataset using data.py. For example: `py data.py -i ./pgns -o ./db.h5`, where pgns is a folder of pgn files (or one massive pgn file, concatenated together with a seperating blank line), and db.h5 is the name of the dataset file. Tested using the lichess database.
- Train with train.py: `py train.py -train_data ./db.h5 -w weights.h5 -o weights.h5 -epochs 10 --ignore_no_weights`, where -w provides the initial weights file, -o provides the output weights, and -epochs tells how long to train for (how many interations of the dataset). --ignore_no_weights means that -w will be ignored if the file does not exist.
- Evaluate the model: `py eval.py ./test.pgn -w ./weights.h5`, where test.pgn is a pgn file for game to be evaluated, and weights.h5 is a file of model weights. The output is an array with probabilities of an outcome. The first is the probability that **whoever just moved** will **lose**.

# Model Inputs:

Bitboard = An 8x8 array of booleans. (1 or 0)

A flattened array of:
- 6 Bitboards representing the positions of each of player one's pieces. (It is always player ones turn, and the boards is always oriented from player ones POV.)
- 6 Bitboards representing the positions of each of player two's pieces.
- 2 Bitboards representing any repetitions that the players have made (Aka what move not to make to avoid a draw by repitition)
- 1 Bitboard representing en-passantable squares.
- A single number representing the ELO of player one. (Normalized)
- A single number representing the ELO of player two (Normalized, like player one)
- A single number representing how many minutes player one has. (Divided by 10, capped at 15)
- A single number representing how many minutes player two has. (Divided by 10, capped at 15)
- 4 booleans (1 or 0) representing whether P1 and P2 can castle, and on what side. (Queen/King)
- Number of moves in which a pawn has not been pushed and a piece has not been captured. After 50 of these moves, the game is drawn. (Also normalized)
