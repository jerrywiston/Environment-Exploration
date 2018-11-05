import numpy as np
from GridMap import *
from MotionModel import *
import random
import math
import utils
import copy
import threading

class Particle:
    def __init__(self, pos, bot_param, gmap, motion):
        self.pos = pos
        self.bot_param = bot_param
        self.gmap = gmap
        self.motion = motion
        self.trajectory = [pos]

    def Sampling(self, aid):
        if aid == 1:
            self.pos = self.motion.Sample(self.pos, self.bot_param[4], 0, 0)
        if aid == 2:
            self.pos = self.motion.Sample(self.pos, -self.bot_param[4], 0, 0)
        if aid == 3:
            self.pos = self.motion.Sample(self.pos, 0, 0, -self.bot_param[5])
        if aid == 4:  
            self.pos = self.motion.Sample(self.pos, 0, 0, +self.bot_param[5])
        
        self.trajectory.append(self.pos)

    def NearestDistance(self, x, y, wsize, th):
        min_dist = 9999
        min_x = None
        min_y = None
        gsize = self.gmap.gsize
        xx = int(round(x/gsize))
        yy = int(round(y/gsize))
        for i in range(xx-wsize, xx+wsize):
            for j in range(yy-wsize, yy+wsize):
                if self.gmap.GetGridProb((i,j)) < th:
                    dist = (i-xx)*(i-xx) + (j-yy)*(j-yy)
                    if dist < min_dist:
                        min_dist = dist
                        min_x = i
                        min_y = j

        return math.sqrt(float(min_dist)*gsize)

    def LikelihoodField(self, sensor_data):
        p_hit = 0.9
        p_rand = 0.1
        sig_hit = 3.0
        q = 1.0
        plist = utils.EndPoint(self.pos, self.bot_param, sensor_data)
        for i in range(len(plist)):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            dist = self.NearestDistance(plist[i][0], plist[i][1], 4, 1.0)
            q = 4 * q * (p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
            #q += math.log(p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
        return q

    def Mapping(self, sensor_data):
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            theta = self.pos[2] + self.bot_param[1] + i*inter
            self.gmap.GridMapLine(
            int(self.pos[0]), 
            int(self.pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta-90))),
            int(self.pos[1]),
            int(self.pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta-90)))
            )

class ParticleFilter:
    def __init__(self, pos, bot_param, gmap, motion, size):
        self.size = size
        self.particle_list = []
        self.weights = np.ones((size), dtype=float) / size
        p = Particle(pos.copy(), bot_param, copy.deepcopy(gmap), motion)
        for i in range(size):
            self.particle_list.append(copy.deepcopy(p))
    
    def ParticleMapping(sensor_data):
        threads = []
        for p in self.particle_list:
            threads.append(threading.Thread(target=p.Mapping, args=(sensor_data,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def Resampling(self, sensor_data):
        map_rec = np.zeros((self.size))
        re_id = np.random.choice(self.size, self.size, p=list(self.weights))
        new_particle_list = []
        for i in range(self.size):
            new_particle_list.append(copy.deepcopy(self.particle_list[re_id[i]]))
        self.particle_list = new_particle_list
        self.weights = np.ones((self.size), dtype=float) / float(self.size)

    def Feed(self, control, sensor_data):
        field = np.zeros((self.size), dtype=float)
        for i in range(self.size):
            self.particle_list[i].Sampling(control)
            field[i] = self.particle_list[i].LikelihoodField(sensor_data)
            self.particle_list[i].Mapping(sensor_data)
        print(field)
        if np.sum(field) != 0:
            self.weights = field / np.sum(field)
        
        Neff = 0
        for i in range(self.weights.shape[0]):
            Neff += self.weights[i]*self.weights[i]
        Neff = 1.0 / Neff
        print(Neff)
        print(self.weights)
        return Neff