import numpy as np
from GridMap import *
import random

class Particle:
    def __init__(self, pos, bot_param, gmap, weight):
        self.pos = pos
        self.bot_param = bot_param
        self.gmap = gmap
        self.weight = weight

    def Sampling(self, control, sig):
        vec = [np.sin(np.deg2rad(self.pos[2])), np.cos(np.deg2rad(self.pos[2]))]
        vel = self.bot_param[4]
        ang = self.bot_param[5]

        if aid == 1:
            self.pos[0] -= vel*vec[0]
            self.pos[1] += vel*vec[1]
        if aid == 2:
            self.pos[0] += vel*vec[0]
            self.pos[1] -= vel*vec[1]
        if aid == 3:
            self.pos[2] -= ang
            self.pos[2] = self.bot_pos[2] % 360
        if aid == 4:  
            self.pos[2] += ang
            self.pos[2] = self.bot_pos[2] % 360
        
        if aid == 5:
            self.pos[1] -= vel
        if aid == 6:
            self.pos[0] -= vel
        if aid == 7:
            self.pos[0] += vel
        if aid == 8:
            self.pos[1] += vel
        
        self.pos[0] += random.gauss(0,sig[0])
        self.pos[1] += random.gauss(0,sig[1])
        self.pos[2] += random.gauss(0,sig[2])

    def LikelihoodField(self, sensor_data):
        pass

    def Mapping(self, sensor_data):
        pass