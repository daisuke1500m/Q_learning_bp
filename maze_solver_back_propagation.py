import numpy as np
import random
import copy
import pprint
import matplotlib.pyplot as plt
import time
import math
from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Policy(Enum):
    EPSILON_GREEDY = 0
    SOFTMAX = 1


class World:
    def __init__(self):
        self.walls = []
        self.maze = [
            "WWWWWWWWWW",
            "WS     W W",
            'W      W W',
            'W WWW    W',
            'W W      W',
            'W W WWWWWW',
            'W W      W',
            'W W  WWWWW',
            'W       GW',
            'WWWWWWWWWW'
        ]
        self.Width = len(self.maze[0])
        self.Height = len(self.maze)
        self.rewards = [[0 for i in range(self.Height)] for l in range(self.Width)]
        self.START = (0, 0)
        self.GOAL = (0, 0)
        self.actions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        for y in range(self.Height):
            for x in range(self.Width):
                if self.maze[y][x] == 'S':
                    self.START = (x, y)
                elif self.maze[y][x] == 'G':
                    self.GOAL = (x, y)
                    self.rewards[x][y] = 1
                elif self.maze[y][x] == 'W':
                    self.walls.append((x, y))
                    self.rewards[x][y] = -1

    # 状態と行動から次の状態を返す関数
    def getNextStatus(self, status, action:Direction):
        # 壁に向かって進む時
        if (status[1] == 0 and action == Direction.UP) \
            or (status[1] == self.Height-1 and action == Direction.DOWN) \
            or (status[0] == 0 and action == Direction.LEFT) \
            or (status[0] == self.Width-1 and action == Direction.RIGHT):
            return None
        
        x,y = status
        if action == Direction.UP:
            y -= 1
        elif action == Direction.DOWN:
            y += 1
        elif action == Direction.LEFT:
            x -= 1
        elif action == Direction.RIGHT:
            x += 1
        return (x, y)

    #　報酬を返す関数
    def R(self, status):
        return self.rewards[status[0]][status[1]]


class Agent:
    def __init__(self, episode=200, epsilon = 0.1, alpha = 0.2, gamma = 0.99, bp = False):
        self.episode_number = episode
        self.world = World()
        self.q_table = np.zeros((self.world.Width, self.world.Height, len(self.world.actions)))
        self.totalQvalue = 0
        self.status = self.world.START
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.learning_progress = []
        self.present_route = []
        self.minimum_step = 100000000
        self.enable_back_propagation = bp
        # 最小のステップとそれにたどりついたエピソード数をタプルで保存
        # (達したエピソード数, ステップ数)
        self.minimum_step_progress = []
        

    def learn(self):
        pre_totalQvalue = self.totalQvalue
        digit_all_episode = int(math.log10(self.episode_number) * 2) + 1

        for i in range(1, self.episode_number):
            print("%d/%d" % (i, self.episode_number), end="")
            self.episode()
            self.learning_progress.append(self.totalQvalue)
            self.minimum_step_progress.append(self.getMinimumRouteStep())
            digit = int(math.log10(i) * 2) + digit_all_episode
            print("\033[%dD" % digit, end="")
        print()


    def episode(self):
        self.status = self.world.START
        visited = []
        while (not self.status == self.world.GOAL):
            action = self.decideActionSoftmax()
            if action == None:
                break

            nextstatus = self.world.getNextStatus(self.status, action)
            
            if nextstatus == None:
                self.q_table[self.status[0]][self.status[1]][action.value] -= 1
                break

            nextmaxQvalue = np.max(self.q_table[nextstatus[0]][nextstatus[1]])

            Qdifference = self.alpha * ( \
                    self.world.R(nextstatus) \
                    + self.gamma * nextmaxQvalue \
                    - self.q_table[self.status[0]][self.status[1]][action.value]\
                                )
            self.q_table[self.status[0]][self.status[1]][action.value] += Qdifference
            self.totalQvalue += Qdifference
            
            visited.append(self.status)
            self.status = nextstatus
        #逆伝搬する部分
        else:
            if len(visited) < self.minimum_step and self.enable_back_propagation:
                self.minimum_step = len(visited)
                for v in reversed(visited):
                    nextmaxQvalue = np.max(self.q_table[self.status[0]][self.status[1]])
                    Qdifference = self.alpha * ( \
                        self.world.R(self.status) \
                        + self.gamma * nextmaxQvalue \
                        - self.q_table[v[0]][v[1]][action.value]\
                                    )
                    self.q_table[v[0]][v[1]][action.value] += Qdifference
                    self.totalQvalue += Qdifference
                    self.status = v
                self.learning_progress.append(self.totalQvalue)

    def decideActionEpsilonGreedy(self):
        if random.random() < self.epsilon:
            # 実施できる行動の中から適当に選ぶ
            return random.choice(self.world.actions)
        else:
            # 行動の中から実施した時のQ値の最も大きい行動を選ぶ
            actionindex = np.argmax(self.q_table[self.status[0]][self.status[1]])
            action = self.world.actions[actionindex]
            return action

    def decideActionSoftmax(self):
        tau = 0.5       

        actionlength = len(self.world.actions)
        denominator = sum([np.exp(self.q_table[self.status[0]][self.status[1]][i] / tau) for i in range(actionlength)])
        probability = [np.exp(self.q_table[self.status[0]][self.status[1]][i] / tau) / denominator for i in range(actionlength)]
        action = np.random.choice(self.world.actions, p=probability)
        return action


    def printSolution(self):
        arrow = ['↑', '↓', '←', '→']

        for y in range(self.world.Height):
            for x in range(self.world.Width):
                if (x,y) in self.world.walls:
                    print('■ ', end='')
                elif (x,y) == self.world.START:
                    print('S ', end='')
                elif (x,y) == self.world.GOAL:
                    print('G ', end='')
                else:
                    index = np.argmax(self.q_table[x][y])
                    print(arrow[index] + ' ', end='')
            print()
        
    def defineRoute(self):
        arrow = ['↑', '↓', '←', '→']
        self.present_route = []

        for y in range(self.world.Height):
            line = ""
            for x in range(self.world.Width):
                if (x,y) in self.world.walls:
                    line = line + "▪️"
                elif (x,y) == self.world.START:
                    line = line + "S"
                elif (x,y) == self.world.GOAL:
                    line = line + "G"
                else:
                    index = np.argmax(self.q_table[x][y])
                    line = line + arrow[index]
            self.present_route.append(line)
    
    # 現在のスタートからゴールまでの最小ステップ数を返す関数
    # Qテーブル通りに辿っていって,
    # ゴールにたどり着いたらそのステップ数を返す
    # 来たことがあるセルにたどりつくか、壁にめり込むか、場外に出る場合にすごく大きい数を返す
    def getMinimumRouteStep(self):
        infinity = 100000000
        status = self.world.START
        move = [(0,-1), (0,1), (-1,0), (1,0)]
        visited = []    

        while(not status == self.world.GOAL):
            visited.append(status)
            x,y = status
            direction_index = np.argmax(self.q_table[x][y])
            x += move[direction_index][0]
            y += move[direction_index][1]
            status = (x, y)
            if status in self.world.walls:
                return infinity
            elif status in visited:
                return infinity
            elif status[0] < 0 or self.world.Width <= status[0] \
                or status[1] < 0 or self.world.Height <= status[1]:
                return infinity
        else:
            return len(visited)
        
    def showQvalueProgress(self):
        index = [i for i in range(len(self.learning_progress))]
        plt.plot(index, self.learning_progress)
        plt.show()

if __name__ == '__main__':
    agent_enable_bp = Agent(episode = 500, bp = True)
    agent_enable_bp.learn()
    agent_disable_bp = Agent(episode = 500, bp = False)
    agent_disable_bp.learn()

    # 学習が終わったら音で教えてくれる
    print("\a")

    agent_enable_bp.printSolution()

    #agent.printSolution()
    #agent.showQvalueProgress()
    
    #　グラフ描画
    data_enable_bp = np.array(agent_enable_bp.learning_progress)
    data_disable_bp = np.array(agent_disable_bp.learning_progress)
    data_enable_step = np.array(agent_enable_bp.minimum_step_progress)
    data_disable_step = np.array(agent_disable_bp.minimum_step_progress)
    t = np.array(list(range(len(agent_enable_bp.learning_progress))))

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)   

    red, blue = "red", "blue"
    label_enable, label_disable = "bp enable", "bp disable"
    ax1.set_xlabel('episode')
    ax1.set_ylabel('progress')
    ax1.set_title('learning progress')
    ax1.grid()
    ax1.legend(loc=0)

    min_dimen = min(len(t), len(data_enable_bp), len(data_disable_bp))
    ax1.plot(t[:min_dimen], data_enable_bp[:min_dimen], color=red, label=label_enable)
    ax1.plot(t[:min_dimen], data_disable_bp[:min_dimen], color=blue, label=label_disable)

    ax2.set_xlabel('episode')
    ax2.set_ylabel('step')
    ax2.set_title('step progress')
    ax2.grid()
    ax2.legend(loc=0)

    min_dimen = min(len(t), len(data_enable_step), len(data_disable_step))
    print(len(t), len(data_enable_step), len(data_disable_step))
    ax2.plot(t[:min_dimen], data_enable_step[:min_dimen], color=red, label=label_enable)
    ax2.plot(t[:min_dimen], data_disable_step[:min_dimen], color=blue, label=label_disable)
    
    fig.tight_layout()
    plt.show()
