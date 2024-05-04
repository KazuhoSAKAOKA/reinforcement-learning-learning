from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from network_common import load_model,NetworkBrain, load_data
from tictactoe_network import MODEL_FILE_BEST, dual_network, DN_FILTERS,DN_INPUT_SHAPE, DN_OUTPUT_SIZE,residual_block,conv,DN_RESIDUAL_NUM, train_cycle_tictactoe, train2_cycle_tictactoe
from network_brain import predict, SelfplayNetworkBrain
from tictactoe_board import TicTacToeBoard
from mini_max import AlphaBetaBrain
from agent import Agent
from game import GameEnv
from self_play import self_play, SelfplayBrain
import numpy as np
import pv_mcts
import pickle
import parameter as PARAM

def output_step(board :TicTacToeBoard, model :Model):
    p, v = predict(model, board)
    print(board)
    print(p)
    print(v)

def degut_predict_only(model):
    board = TicTacToeBoard()
    output_step(board, model)

    board, _ = board.transit_next(4)
    output_step(board, model)

    board, _ = board.transit_next(1)
    output_step(board, model)

    board, _ = board.transit_next(2)
    output_step(board, model)

    board, _ = board.transit_next(6)
    output_step(board, model)

    board, _ = board.transit_next(5)
    output_step(board, model)


def output_mcts_step(board :TicTacToeBoard, brain: NetworkBrain):
    board = step(board, [4,1,2,6])
    v = brain.select_action(board)
    print(board)
    print(brain.get_last_policies())

def debug_predict(_1, board : TicTacToeBoard):
    legal_actions = board.get_legal_actions()
    p = 1.0 / len(legal_actions)
    return np.full(shape=(len(legal_actions),), fill_value=p), 0.0
def step(board :TicTacToeBoard, steps : list) -> TicTacToeBoard:
    for action in steps:
        board, _ = board.transit_next(action)
    return board

def debug_mcts(model):
    board = TicTacToeBoard()
    brain = NetworkBrain(0.1, 5000, lambda x: debug_predict(model, x), lambda x: debug_predict(model, x))
    #output_mcts_step(board, brain)

    board = step(board, [4,1,2,6])
    p = predict(model, board)
    print(p)
    output_mcts_step(board, brain)

def debug_play(model):
    board = TicTacToeBoard()
    brain_agent = Agent(NetworkBrain(0, 50, lambda x: predict(model, x), lambda x: predict(model, x)))
    alpha_beta_agent = Agent(AlphaBetaBrain())
    #env = GameEnv(board, brain_agent, alpha_beta_agent)
    #env.play_n(5)
    env = GameEnv(board, alpha_beta_agent, brain_agent)
    env.play_n(5)

def load_history_file(history_file):
    with open(history_file, mode='rb') as f:
        return pickle.load(f)
    

def history_convert(history: str):
    data = load_history_file(history)
    with open('./history.txt', 'w') as f:
        for x in data:
            f.write(str(x))
            f.write('\n')

def train_handdata():
    dual_network()
    model = load_model(MODEL_FILE_BEST)
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    empty_board = TicTacToeBoard()
    one_board,_ = empty_board.transit_next(4)
    p, v = predict(model, empty_board)
    print(p)
    print(v)
    p, v = predict(model, one_board)
    print(p)
    print(v)
    #xs = np.zeros(shape=(1,3,3,2), dtype=np.int8)
    xs = np.array([
                    [
                        [
                            [0,0],[0,0],[0,0]
                        ] ,
                        [
                            [0,0],[0,1],[0,0]
                        ] ,
                        [	
                            [0,0],[0,0],[0,0]
                        ] ,
                    ],
                ],
                    dtype=np.int8)
    print(xs)
    print(xs.shape)

    y_policies = np.full(shape=(1,9), fill_value=0.11)
    y_values = np.zeros(shape=(1,1), dtype=np.float32)

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)
    epoch_count = 1000
    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print('\rTrain {}/{}'.format(epoch + 1, epoch_count), end=''))

    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=epoch_count, 
              verbose=0, callbacks=[lr_decay, print_callback])
    print()
    p, v = predict(model, empty_board)
    print(p)
    print(v)
    p, v = predict(model, one_board)
    print(p)
    print(v)
#file = MODEL_FILE_BEST
#file = './model/tictactoe/20240314225839.keras'
#model = load_model(MODEL_FILE_BEST)
#debug_play(model)

#degut_predict_only(model)
#debug_mcts(model)


#K.clear_session()
#del model

#train_handdata()

#history_convert('./data/tictactoe/second/20240405234236.history')


def test1():
    PARAM.alpha = 0.0
    board = TicTacToeBoard()
    first_model = load_model('./model/tictactoe/first/20240410073402.keras')
    second_model = load_model('./model/tictactoe/second/20240410073806.keras')
    network_agent = Agent(NetworkBrain(0, 50, lambda x: predict(second_model, x), lambda x: predict(first_model, x)))

    board, _ = board.transit_next(4)
    print(board)
    board, _ = board.transit_next(0)
    print(board)
    board, _ = board.transit_next(1)
    print(board)
    #selected = network_agent.brain.select_action(board)
    #print(selected)
    y = second_model.predict(board.reshape_to_input(), batch_size=1, verbose=0)
    print(y)


def train_hand_data():
    input = Input(shape=DN_INPUT_SHAPE)

    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    #x = GlobalAveragePooling2D()(x)

    p = conv(DN_FILTERS)(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = GlobalAveragePooling2D()(p)
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005), activation='softmax',name='pi')(p)

    v = conv(DN_FILTERS)(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = GlobalAveragePooling2D()(v)
    v = Dense(1, kernel_regularizer=l2(0.0005))(v)
    v = Activation('tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p,v])
    model.summary()
    
    def add(x,a):
        return np.append()

    board = TicTacToeBoard()
    board, _ = board.transit_next(4)
    board, _ = board.transit_next(0)
    board, _ = board.transit_next(1)



    xs = board.reshape_to_input()
    y = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]])
    v = np.array([0.8])
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)
    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print('\rTrain {}/{}'.format(epoch + 1, 200), end=''))
    model.fit(xs, [[y], [v]], batch_size=128, epochs=200, verbose=0, callbacks=[lr_decay, print_callback])
    print("complete training")
    pv = model.predict(xs, batch_size=1, verbose=0)
    print(pv)

#train_hand_data()
#train_cycle_tictactoe(brain_evaluate_count=10, selfplay_repeat=10, epoch_count=10, cycle_count=10, eval_count=10)
#train2_cycle_tictactoe(500, 200, 5, 5, 0.1)
train_cycle_tictactoe(5, 2, 2, 1, 0)

#dual_network(MODEL_FILE_BEST)
#model = load_model(MODEL_FILE_BEST)

#first = SelfplayNetworkBrain(10, model)
#second = SelfplayNetworkBrain(10, model)
#self_play(first, second, TicTacToeBoard(), 1, './data/tictactoe')

