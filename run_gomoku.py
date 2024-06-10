from gomoku_network import train_cycle_gomoku,create_network_parameter_dual,create_selfplay_parameter,create_brain_parameter, create_initial_selfplay_parameter
from parameter import NetworkParameter,SelfplayParameter,BrainParameter,ExplorationParameter,NetworkType,HistoryUpdateType,ActionSelectorType, InitSelfplayParameter, judge_stats

BOARD_SIZE=11


n_param = create_network_parameter_dual(BOARD_SIZE)
s_param = create_selfplay_parameter(BOARD_SIZE)
b_param = create_brain_parameter(BOARD_SIZE)
i_param = create_initial_selfplay_parameter(BOARD_SIZE)


s_param.selfplay_repeat = 100
s_param.train_epoch = 200
s_param.evaluate_count = 10

b_param.mcts_evaluate_count = 150

i_param.selfplay_repeat = 1000
i_param.train_epoch = 500

train_cycle_gomoku(
                board_size=BOARD_SIZE,
                network_param=n_param,
                selfplay_param=s_param,
                brain_param=b_param,
                exploration_param=ExplorationParameter(),
                initial_selfplay_param=i_param)

