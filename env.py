import numpy as np
# from gym.spaces import Discrete, Tuple
import matplotlib.pyplot as plt

# 状态空间
class State():
    def __init__(self,Dn,remaining_frame):
        self.Dn = Dn
        self.tolerable_frame = None
        self.h = None
        self.remaining_frame = remaining_frame

    def to_numpy(self):
        # 将所有属性转换为 NumPy 数组
        # Dn_array = np.array(self.Dn, dtype=np.float32)
        tolerable_frame_array = np.array(self.tolerable_frame,
                                         dtype=np.float32) if self.tolerable_frame is not None else np.array([])
        h_array = np.array(self.h, dtype=np.float32) if self.h is not None else np.array([])
        remaining_frame_array = np.array(self.remaining_frame,
                                         dtype=np.float32) if self.remaining_frame is not None else np.array([])

        # 将所有属性展平并连接成一个数组
        return np.concatenate([
            # Dn_array.flatten(),
            tolerable_frame_array.flatten(),
            h_array.flatten(),
            remaining_frame_array.flatten()
        ])

    def __repr__(self):
        return f"<State: [{self.Dn}, {self.tolerable_frame}, {self.h}, {self.remaining_frame}]>"

    def clone(self):
        return State(self.Dn, self.tolerable_frame, self.h, self.remaining_frame)

    def __hash__(self):
        return hash((self.Dn, self.tolerable_frame, self.h, self.remaining_frame))

    def __eq__(self, other):
        return self.Dn == other.Dn and self.tolerable_frame == other.tolerable_frame and self.h == other.h and self.remaining_frame == other.remaining_frame  

# 动作空间
class Action():
    def __init__(self, N, M):
        # 创建一个N*(M+1)的动作空间矩阵，每行代表一个动作空间，每个动作空间有M+1个可能的动作
        self.action_space = [list(range(M + 1)) for _ in range(N)]

    # 返回动作空间
    def get_action_space(self):
        return self.action_space
    
    # 返回选择的动作
    def pop_actions(self):
        # 从每个动作空间（即矩阵的每一行）中随机选择一个动作
        actions = [np.random.choice(action_space) for action_space in self.action_space]
        return actions
    
    # 返回动作对应的标签，如Action_0, Action_1, ...
    def pop_labels(self):
        # 从每个动作空间（即矩阵的每一行）中随机选择一个动作
        actions = [np.random.choice(action_space) for action_space in self.action_space]
        labels = []
        for action in actions:
            label = f'Action_{action}'
            labels.append(label)
        return labels

class Environment():
    def __init__(self, N, M, Width, fv, Dn, Cn, fn, distance, P, mu, eta, sigma, alpha, T, threshold, battery=10000):
        # 中间变量
        self.N = N
        self.M = M
        self.Width = Width
        self.fv = fv
        self.Dn = Dn
        self.Cn = Cn
        self.fn = fn
        self.distance = distance
        self.P = P
        self.mu = mu
        self.eta = eta
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.threshold = threshold
        self.state = State(self.Dn, self.T)
        self.action = Action(self.N, self.M)
        self.reward = 0
        self.battery = battery
        # 要存储的变量
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = [] # 用于存储log(π(a|s))
        self.is_terminals = []
        # step要用的变量
        self.e_local = np.zeros(self.N)  # 累计所有用户的本地能耗
        self.e_cycle = np.zeros(self.N)
        self.T_success = np.ceil(np.random.rand(self.N) * self.T) # 每个用户所需求的“成功传输的帧数”不同，比如打游戏、看电视的用户对延迟要求不同, 取整
        self.tolerable_frame = self.T - self.T_success # 每个用户所能“容忍的帧数”

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def reset(self, seed = None):  # sourcery skip: class-extract-method
        if seed is not None:
            np.random.seed(seed)

        self.reward = 0
        self.done = False
        self.tolerable_frame = self.T - self.T_success # 每个用户所能“容忍的帧数”
        self.gn = np.random.rayleigh(1, (self.N, self.M + 1)) # 瑞利衰减系数,按照瑞利分布随机取的值
        self.distant = np.random.uniform(low=0, high=200, size=(self.N, 1)) # 随机取用户到边缘服务器的距离，待修改
        self.h = self.gn * self.distant ** (-self.alpha) # 计算信道增益
        self.state.h = self.h
        self.state.tolerable_frame = self.tolerable_frame
        return self.state.to_numpy()

    ## 参数：动作action，一次选择的所有用户的动作
    ## 返回 下一个状态next_state，奖励reward，是否结束done
    def step(self, action = None, seed = None):
        if seed is not None:
            np.random.seed(seed)
        # sourcery skip: sum-comprehension
        I = np.zeros(self.N) # 判断延迟是否满足最大延迟要求，不满足则为1，满足则为0 


        self.gn = np.random.rayleigh(1, (self.N, self.M + 1)) # 瑞利衰减系数,按照瑞利分布随机取的值
        self.distant = np.random.uniform(low=0, high=200, size=(self.N, 1)) # 随机取用户到边缘服务器的距离，待修改
        self.h = self.gn * self.distant ** (-self.alpha) # 计算信道增益
        self.state.h = self.h

        action_choose = self.action.pop_actions() if action is None else action
        self.actions.append(action_choose) # 记录动作

        h_choose = np.zeros(self.N) # 选择动作对应的信道增益
        rate = np.zeros(self.N)
        delay = np.zeros(self.N)

        # 按照选择的动作，取出对应的信道增益, n是目前第几个人， action是一个列表，每个元素是一个用户选择的动作
        # rate包括每个用户的速率
        # print(type(action_choose))
        # print(action_choose)
        for n, action in enumerate(action_choose): 
            # 用户选择的通道不是本地计算
            if action != 0:
                h_choose[n] = self.h[n][action]
                Pn_all = 0
                # 一个信道的全部用户的速率和
                for i in range(n+1, self.N):
                    Pn_all += self.P[i] * (h_choose[n] ** 2)
                rate_n = self.Width * np.log2( 1 + ( self.P[n] * h_choose[n] ** 2 / ( Pn_all + self.Width * self.sigma ** 2 ) ) ) # 计算速率
                rate[n] = rate_n
                # 计算延迟
                delay[n] = self.Dn[n] * self.Cn[n] / self.fv + self.Dn[n] / rate[n]
            # 用户选择的通道是本地计算
            else:
                rate[n] = 0
                delay[n] = self.Dn[n] * self.Cn[n] / self.fn[n]
                # 计算功耗
                self.e_cycle[n] = self.eta * self.fn[n] ** 2
                self.e_local[n] = self.mu[n] * self.Dn[n] * self.Cn[n] * self.e_cycle[n]
        ## 至此，得到了 delay，rate，e_local，e_cycle均为N维向量，分别代表每个用户的延迟，速率，本地能耗，计算能耗

        self.reward += (self.battery - np.sum(self.e_local) ) * 1e-4 # 计算本地能耗
        for N in range(self.N):
            if delay[N] > self.threshold:
                I[N] = 1
                self.reward -= 1
            else:
                I[N] = 0

        self.tolerable_frame = self.tolerable_frame - I
        self.state.tolerable_frame = self.tolerable_frame # 存储当前状态的tolerable_frame
        for N in range(self.N):
            if self.tolerable_frame[N] == 0:
                self.done = True
                self.reward -= 10000
                break
            else:
                self.done = False

        self.states.append(self.state) # 记录状态
        self.rewards.append(self.reward) # 记录奖励
        self.is_terminals.append(self.done) # 记录是否结束

        return self.state.to_numpy(), self.reward, self.done

    def render(self):
        # 可视化当前环境
        plt.plot(self.rewards)
        plt.title('Rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.show()

    def is_done(self):
        # 判断当前环境是否结束
        return bool(self.is_terminals[-1])


## 全局变量
    # N：VU的数量
    # M：边缘服务器的数量
    # Width 带宽 10 MHz(注：修改为180 KHz)
    # fv 边缘服务器总计算能力
    # Dn, Cn 任务量大小，所需cpu周期数, (300~500kb), (900, 1100)兆周期数 1Mhz = 1000khz = 1000*1000hz
    # fn 用户本地计算能力 1GHz/s | [0.5, 1.5]GHz/s (1000*1000*1000)
    # distance 用户距离
    # P 传输功率 | mW (毫瓦)
    # alpha 瑞利衰减参数
    # mu 能量加权参数 eta 能量参数
    # sigma 噪声方差 alpha 瑞利衰减参数
    # T 共T帧
    # threshold 延迟阈值
    # battery 电池容量
if __name__ == '__main__':
    N = 3
    M = 3
    Width = 180
    fv = 20
    Dn , Cn = np.random.uniform(300, 500, N), [600 for _ in range(N)]
    fn = np.random.uniform(0.5, 1.5, N)
    distance = np.random.uniform(0, 200, N)
    P = np.random.uniform(40000, 60000, N)
    mu , eta = np.zeros(N) , 0.5
    sigma , alpha = 1e-2 , 2
    threshold = 50000
    T = 90
    battery = 10000

    env = Environment(N, M, Width, fv, Dn, Cn, fn, distance, P, mu, eta, sigma, alpha, T, threshold, battery)
    for _ in range(10):
        next_state, reward, done = env.step()
        #print(env.h)
        if done:
            break
    env.render()

    # print(env.states)
    # print(env.actions)
    # print(env.rewards)
    # print(env.is_terminals)
    # env.clear_memory() 
