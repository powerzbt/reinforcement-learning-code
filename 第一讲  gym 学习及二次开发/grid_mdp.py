

import logging
import numpy
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)
#?

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = [1,2,3,4,5,\
                       6,7,8,9,10,\
                       11,12,13,14,15,\
                       16,17,18,19,20,\
                       21,22,23,24,25] #状态空间
        
        self.x=[100,200,300,400,500,\
               100,200,300,400,500,\
               100,200,300,400,500,\
               100,200,300,400,500,\
               100,200,300,400,500]
            #origin : leftbotton corner
            
        self.y=[500,500,500,500,500,\
               400,400,400,400,400,\
               300,300,300,300,300,\
               200,200,200,200,200,\
               100,100,100,100,100]
        
        self.terminate_states = dict()  #终止状态为字典格式
        self.terminate_states[15] = 1


        self.actions = ['n','e','s','w']

        self.rewards = dict();        #回报的数据结构为字典
        self.rewards['10_s'] = 1.0
        self.rewards['14_e'] = 1.0
        self.rewards['20_n'] = 1.0

        self.t = dict();             #状态转移的数据格式为字典
        self.t['1_e'] = 2
        self.t['1_s'] = 6
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['2_s'] = 7
        self.t['3_s'] = 8
        self.t['3_w'] = 2
        self.t['5_s'] = 10
        self.t['6_n'] = 1
        self.t['6_e'] = 7
        self.t['7_w'] = 6
        self.t['7_n'] = 2
        self.t['7_e'] = 8
        self.t['8_w'] = 7
        self.t['8_n'] = 3
        self.t['8_s'] = 13
        self.t['10_n'] = 5
        self.t['10_s'] = 15
        self.t['13_n'] = 8
        self.t['13_e'] = 14
        self.t['13_s'] = 18
        self.t['14_e'] = 15
        self.t['14_s'] = 19
        self.t['14_w'] = 13
        self.t['15_s'] = 20
        self.t['15_w'] = 14
        self.t['15_n'] = 10
        self.t['16_e'] = 17
        self.t['16_s'] = 21
        self.t['17_e'] = 18
        self.t['17_s'] = 22
        self.t['17_w'] = 16
        self.t['18_w'] = 17
        self.t['18_s'] = 13
        self.t['18_e'] = 19
        self.t['19_w'] = 18
        self.t['19_n'] = 14
        self.t['19_e'] = 20
        self.t['20_w'] = 19
        self.t['20_n'] = 15
        self.t['21_n'] = 16
        self.t['21_e'] = 22
        self.t['22_w'] = 21
        self.t['22_n'] = 17

        

        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions
    def getTerminate_states(self):
        return self.terminate_states
    def setAction(self,s):
        self.state=s

    def step(self, action):
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
    def reset(self):
        statelist = [1,2,3,5,\
                    6,7,8,10,\
                    13,14,15,\
                    16,17,18,19,20,\
                    21,22]
        self.state = random.choice(statelist)
        return self.state
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 750
        screen_height = 750

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界
            self.line11 = rendering.Line((50,50),(50,550))
            self.line12 = rendering.Line((150,50),(150,550))
            self.line13 = rendering.Line((250,50),(250,550))
            self.line14 = rendering.Line((350,50),(350,550))
            self.line15 = rendering.Line((450,50),(450,550))
            self.line16 = rendering.Line((550,50),(550,550))
            
            self.line21 = rendering.Line((50,50),(550,50))
            self.line22 = rendering.Line((50,150),(550,150))
            self.line23 = rendering.Line((50,250),(550,250))
            self.line24 = rendering.Line((50,350),(550,350))
            self.line25 = rendering.Line((50,450),(550,450))
            self.line26 = rendering.Line((50,550),(550,550))
            #创建第一个墙
            wall1 = rendering.FilledPolygon([(350,350), (350,550), (450,550), (450,350)])
            wall1.set_color(0,0,0)
            self.wall1=wall1
            #创建第二个墙
            wall2 = rendering.FilledPolygon([(50,250), (50,350), (250,350), (250,250)])
            wall2.set_color(0,0,0)
            self.wall12=wall2
            #创建第三个墙
            wall3 = rendering.FilledPolygon([(250,50), (250,150), (550,150), (550,50)])
            wall3.set_color(0,0,0)
            self.wall3=wall3
            #创建机器人
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)
            self.line13.set_color(0, 0, 0)
            self.line14.set_color(0, 0, 0)
            self.line15.set_color(0, 0, 0)
            self.line16.set_color(0, 0, 0)
            
            self.line21.set_color(0, 0, 0)
            self.line22.set_color(0, 0, 0)
            self.line23.set_color(0, 0, 0)
            self.line24.set_color(0, 0, 0)
            self.line25.set_color(0, 0, 0)
            self.line26.set_color(0, 0, 0)

            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.line13)
            self.viewer.add_geom(self.line14)
            self.viewer.add_geom(self.line15)
            self.viewer.add_geom(self.line16)
            self.viewer.add_geom(self.line21)
            self.viewer.add_geom(self.line22)
            self.viewer.add_geom(self.line23)
            self.viewer.add_geom(self.line24)
            self.viewer.add_geom(self.line25)
            self.viewer.add_geom(self.line26)
            
            self.viewer.add_geom(self.wall1)
            self.viewer.add_geom(self.wall2)
            self.viewer.add_geom(self.wall3)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])



        return self.viewer.render(return_rgb_array=mode == 'rgb_array')







