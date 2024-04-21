# パラメータの準備
PV_EVALUATE_COUNT = 100 # 1推論あたりのシミュレーション回数（本家は1600）
#C_PUCT = 2.5
C_BASE = 19652
C_INIT = 1.25
ALPHA = 0 # 0.3
EPSILON = 0 #0.25
GAMMA = 0.99

class Parameter:
    def __init__(self, c_base : float, c_init : float, alpha : float, epsilon : float, evaluate_count : int = PV_EVALUATE_COUNT, gamma : float = GAMMA):
        self.c_base = c_base
        self.c_init = c_init
        self.alpha = alpha
        self.epsilon = epsilon
        self.evaluate_count = evaluate_count
        self.gamma = gamma

PARAM = Parameter(C_BASE, C_INIT, ALPHA, EPSILON, PV_EVALUATE_COUNT, GAMMA)
