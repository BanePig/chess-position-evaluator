# Chess Position Evaluator

Hackathon project to evaluate a chess position using tensorflow, and a lot of data.

# Usage:

**YOU MUST USE PYTHON 3.6-3.8 TO INSTALL TENSORFLOW (AND THUS USE THE PROGRAM)**

- Install the requirements with pip install -r requirements.txt (From inside repository directory)
- Other steps coming soon

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
