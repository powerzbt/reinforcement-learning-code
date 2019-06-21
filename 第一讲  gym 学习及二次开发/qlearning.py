import sys
import gym
import random
random.seed(0)
import time
import matplotlib.pyplot as plt

grid = gym.make('GridWorld-v0')     #grid is a instance of class GridEnv
#grid=env.env                     #创建网格世界
states = grid.env.getStates()        #获得网格世界的状态空间. available states
actions = grid.env.getAction()      #获得网格世界的动作空间
gamma = grid.env.getGamma()       #获得折扣因子
#计算当前策略和最优策略之间的差
best = dict() #储存最优行为值函数        
#?
def read_best():    #obtain the best qfunction by reading a file, actual best qfunction, used to plot graph at last step, for 
                        #comparition only, not used in policy iteration
    f = open("best_qfunc")
    
                                                        #best_qfunc:
                                                        '''
                                                        1_n:0.512000
                                                        1_e:0.640000
                                                        1_s:-1.000000
                                                        1_w:0.512000
                                                        2_n:0.640000
                                                        2_e:0.800000
                                                        2_s:0.640000
                                                        2_w:0.512000
                                                        3_n:0.800000
                                                        3_e:0.640000
                                                        3_s:1.000000
                                                        3_w:0.640000
                                                        4_n:0.640000
                                                        4_e:0.512000
                                                        4_s:0.640000
                                                        4_w:0.800000
                                                        5_n:0.512000
                                                        5_e:0.512000
                                                        5_s:-1.000000
                                                        5_w:0.640000
                                                        6_n:0.000000
                                                        6_e:0.000000
                                                        6_s:0.000000
                                                        6_w:0.000000
                                                        7_n:0.000000
                                                        7_e:0.000000
                                                        7_s:0.000000
                                                        7_w:0.000000
                                                        8_n:0.000000
                                                        8_e:0.000000
                                                        8_s:0.000000
                                                        8_w:0.000000
                                                        '''
        
    for line in f:      #per line as well as empty line between two lines:
                                                        '''
                                                        >>> f = open("best_qfunc")
                                                        >>> for line in f:
                                                        ...     print(line)
                                                        ... 
                                                        1_n:0.512000

                                                        1_e:0.640000

                                                        1_s:-1.000000

                                                        1_w:0.512000

                                                        2_n:0.640000

                                                        2_e:0.800000

                                                        2_s:0.640000

                                                        2_w:0.512000

                                                        3_n:0.800000
                                                        ...

                                                        '''
        line = line.strip()     #strip by "/n"
        if len(line) == 0: continue     #do not process empty line
        eles = line.split(":")    #sepreat the line to key and value
        best[eles[0]] = float(eles[1])      '''add to dictionary called best which stores the best action value of all states
                                            using dict[key]=value'''
#计算值函数的误差                   #sum of squared errors between current qfunction and the best qfunction
                                 #qfunction: q(S,A);        a dictionary where S_A is key, q(S,A) is value
def compute_error(qfunc):        #input:a qfunction (a dictionary)    output:sum of squared errors
    sum1 = 0.0
    for key in qfunc:
        error = qfunc[key] -best[key]
        sum1 += error *error
    return sum1

#  贪婪策略
def greedy(qfunc, state):  '''input:q(S,A) of all S_A; s'
                                        a qfunction (a dictionary containing q(S,A) of all S and all A) and state (a certain S')
                              output:argmax_a q(s',a)
                                            the best action of s' according to greedy '''
    amax = 0            #the index of best action for s'
    
    #regard the first action as best action, then compare with others
    key = "%d_%s" % (state, actions[0])         # s'_a0
    qmax = qfunc[key]                           # value of qfunc[s'_a0], i.e. q(s',a0)
    for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数
        key = "%d_%s" % (state, actions[i])     # s'_ai
        q = qfunc[key]                          # qfunc[s'_ai], i.e. q(s',ai)
        if qmax < q:
            qmax = q
            amax = i                            #index of action corresponding to qmax
    return actions[amax]                        #return action that give maximum action value to s'


#######epsilon贪婪策略
def epsilon_greedy(qfunc, state, epsilon):          '''input:q(S,A) of all S_A; s; ε
                                                       output:action a
                                                            the selected action of s according to ε greedy '''
    amax = 0                                    #index of a
    key = "%d_%s"%(state, actions[0])           # s_a0
    qmax = qfunc[key]                           # value of qfunc[s_a0], i.e. q(s,a0)
    
    for i in range(len(actions)):    '''#扫描动作空间得到最大动作值函数,which may be used later with probability 1-ε
                                            no matter used it or not, just calculate it first'''
        key = "%d_%s"%(state, actions[i])       # s_ai
        q = qfunc[key]                          # qfunc[s_ai], i.e. q(s,ai)
        if qmax < q:
            qmax = q
            amax = i
            
    #概率部分
    pro = [0.0 for i in range(len(actions))]    #pro = [0.0, 0.0, ..., 0.0]
                                                    ##of 0.0 = len(actions)
    pro[amax] += 1-epsilon                      #greedy action:pro=1-ε
    for i in range(len(actions)):               #other actions:pro=ε/n
        pro[i] += epsilon/len(actions)

    ##选择动作
    r = random.random()                         #float number lies in 0 to 1
    s = 0.0
    for i in range(len(actions)):               
        s += pro[i]
        if s>= r: return actions[i]
    return actions[len(actions)-1]              #return the action upto which the accumulate probability lager than random # r

def qlearning(num_iter1, alpha, epsilon):       #input n; α; ε
                                                    #n is iteration depth 
    x = []
    y = []
    qfunc = dict()   #行为值函数为字典
    #初始化行为值函数为0
    for s in states:                            #set all q(S,A) as 0
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
    for iter1 in range(num_iter1):              #number of state-initiallization times. i.e. how many chains are selected
        x.append(iter1)
        y.append(compute_error(qfunc))          #very large now

        #初始化初始状态
        s = grid.reset()                        #s=states[int(random.random() * len(self.states))]
                                                    #randomly select one from 1 to 8
        a = actions[int(random.random()*len(actions))]
                                                    #randomly select action from ['n','e','s','w']
        t = False                                   #terminal or not
        count = 0                                   #in each while loop:
                                                        #count of "state go on depth"
                                                        #after line 210, value iteration, s = s1, go on next state point
                                                    #each for loop, the "state go on depth" same, each chain same length
        while False == t and count <100:
            key = "%d_%s"%(s, a)
            #与环境进行一次交互，从环境中得到新的状态及回报
            s1, r, t1, i =grid.step(a)
                                                '''
                                                    def step(self, action):     #input: a
                                                                                    #user assigned action
                                                                                #output: next_state, reward, is_terminal,{}
                                                                                
                                                        #系统当前状态
                                                            state = self.state
                                                            if state in self.terminate_states:
                                                                return state, 0, True, {}
                                                            key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值

                                                            #状态转移
                                                            if key in self.t:
                                                                next_state = self.t[key]
                                                            else:
                                                                next_state = state
                                                            self.state = next_state

                                                            is_terminal = False

                                                            if next_state in self.terminate_states:
                                                                is_terminal = True

                                                            if key not in self.rewards:
                                                                r = 0.0
                                                            else:
                                                                r = self.rewards[key]


                                                            return next_state, r,is_terminal,{}
                                                '''
            key1 = ""
            #s1处的最大动作
            a1 = greedy(qfunc, s1)
            key1 = "%d_%s"%(s1, a1)
            #利用qlearning方法更新值函数
            qfunc[key] = qfunc[key] + alpha*(r + gamma * qfunc[key1]-qfunc[key])
            #转到下一个状态
            s = s1
            a = epsilon_greedy(qfunc, s1, epsilon)
            count += 1
    plt.plot(x,y,"-.,",label ="q alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    plt.show()              #show the plot
    return qfunc

read_best()     
qlearning(100, 0.8, 0.1)        #test 



        

