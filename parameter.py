# パラメータ初期


#C_PUCT = 2.5
C_BASE = 19652
C_INIT = 1.25

# ディリクレノイズ
ALPHA = 0.3
EPSILON = 0.25

# 割引率
GAMMA = 0.99
# ボルツマン定数温度
TEMPERATURE = 1.0
class Parameter:
    def __init__(self, c_base : float, c_init : float, alpha : float, epsilon : float, gamma : float = GAMMA, temperature : float = TEMPERATURE):
        self.c_base = c_base
        self.c_init = c_init
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.temperature = TEMPERATURE

PARAM = Parameter(c_base=C_BASE
                  ,c_init= C_INIT
                  ,alpha= ALPHA
                  ,epsilon= EPSILON
                  ,gamma= GAMMA
                  ,temperature= TEMPERATURE)
