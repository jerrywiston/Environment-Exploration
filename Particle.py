import numpy as np
from GridMap import *
import random

class Particle:
    def __init__(self, pos, bot_param, gmap):
        self.pos = pos
        self.bot_param = bot_param
        self.gmap = gmap

    def Sampling(self, aid, sig):
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
            self.pos[2] = self.pos[2] % 360
        if aid == 4:  
            self.pos[2] += ang
            self.pos[2] = self.pos[2] % 360
        
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
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            theta = self.pos[2] + self.bot_param[1] + i*inter
            self.gmap.GridMapLine(
            int(self.pos[0]), 
            int(self.pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
            int(self.pos[1]),
            int(self.pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
            )
    