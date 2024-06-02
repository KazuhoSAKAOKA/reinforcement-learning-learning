import tensorflow as tf
from tictactoe_board import TicTacToeBoard

if __name__ == '__main__':
    model = tf.keras.models.load_model('/home/kazuho/python/reinforcement-learning-learning/test_files/tictactoe_network/20240602154016.keras')
    game_board = TicTacToeBoard(3)
    x = game_board.reshape_to_input()
    y = model.predict(x)
    print(y)
