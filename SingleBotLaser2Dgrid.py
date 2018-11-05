import numpy as np
import random
import utils
import cv2
import math
from GridMap import *
from ParticleFilter import *
import copy
import matplotlib.pyplot as plt
import Icp2d
from MotionModel import *

class SingleBotLaser2D:
    def __init__(self, bot_pos, bot_param, fname, motion):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.img_map = self.Image2Map(fname)
        self.motion = motion

        scale = 1
        img = self.Image2Map(fname)
        img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.img_map = img
    
    def BotAction(self, aid):
        if aid == 1:
            self.bot_pos = self.motion.Sample(self.bot_pos, self.bot_param[4], 0, 0)
        if aid == 2:
            self.bot_pos = self.motion.Sample(self.bot_pos, -self.bot_param[4], 0, 0)
        if aid == 3:
            self.bot_pos = self.motion.Sample(self.bot_pos, 0, 0, -self.bot_param[5])
        if aid == 4:  
            self.bot_pos = self.motion.Sample(self.bot_pos, 0, 0, self.bot_param[5])

    def Sensor(self):
        sense_data = []
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            theta = self.bot_pos[2] + self.bot_param[1] + i*inter
            sense_data.append(self.RayCast(np.array((self.bot_pos[0], self.bot_pos[1])), theta))
        return sense_data

    def RayCast(self, pos, theta):
        end = np.array((pos[0] + self.bot_param[3]*np.cos(np.deg2rad(theta-90)), pos[1] + self.bot_param[3]*np.sin(np.deg2rad(theta-90))))

        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = utils.Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.bot_param[3]
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.6:
                tmp = math.pow(float(p[0]) - pos[0], 2) + math.pow(float(p[1]) - pos[1], 2)
                tmp = math.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist

    def Image2Map(self, fname):
        im = cv2.imread(fname)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m

def AdaptiveGetMap(gmap):
    mimg = gmap.GetMapProb(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20 )
    #mimg = gmap.GetMapProb(0,500,0,500)
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg

def SensorData2PointCloud(sensor_data, bot_pos, bot_param):
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    tmp = []
    for i in range(len(sensor_data)):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        tmp.append(plist[i])
    tmp = np.array(tmp)
    return tmp
 
def Rotation2Deg(R):
    cos = R[0,0]
    sin = R[1,0]
    theta = np.rad2deg(np.arccos(np.abs(cos)))
    
    if cos>0 and sin>0:
        return theta
    elif cos<0 and sin>0:
        return 180-theta
    elif cos<0 and sin<0:
        return 180+theta
    elif cos>0 and sin<0:
        return 360-theta
    elif cos==0 and sin>0:
        return 90.0
    elif cos==0 and sin<0:
        return 270.0
    elif cos>0 and sin==0:
        return 0.0
    elif cos<0 and sin==0:
        return 180.0

