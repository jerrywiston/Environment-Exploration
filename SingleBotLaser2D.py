import numpy as np
import cv2
from GridMap import *

class SingleBotLaser2D:
    def __init__(self, bot_pos, bot_param):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.line_list = []

    def BotAction(self, aid):
        vec = [np.sin(np.deg2rad(self.bot_pos[2])), np.cos(np.deg2rad(self.bot_pos[2]))]
        vel = self.bot_param[4]
        ang = self.bot_param[5]
        # front
        if aid == 1:
            self.bot_pos[0] -= vel*vec[0]
            self.bot_pos[1] += vel*vec[1]
        if aid == 2:
            self.bot_pos[0] += vel*vec[0]
            self.bot_pos[1] -= vel*vec[1]
        if aid == 3:
            self.bot_pos[2] -= ang
            self.bot_pos[2] = self.bot_pos[2] % 360
        if aid == 4:  
            self.bot_pos[2] += ang
            self.bot_pos[2] = self.bot_pos[2] % 360
        
        if aid == 5:
            self.bot_pos[1] -= vel
        if aid == 6:
            self.bot_pos[0] -= vel
        if aid == 7:
            self.bot_pos[0] += vel
        if aid == 8:
            self.bot_pos[1] += vel

    def Sensor(self):
        sense_data = []
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(bot_param[0]):
            theta = self.bot_pos[2] + self.bot_param[1] + i*inter
            sense_data.append(self.RayCast(np.array((self.bot_pos[0], self.bot_pos[1])), theta))
        return sense_data

    def RayCast(self, pos, theta):
        p = np.array((pos[0],pos[1]))
        max_dist = self.bot_param[3]
        r = np.array((max_dist*np.cos(np.deg2rad(theta)), max_dist*np.sin(np.deg2rad(theta))))
        dist = np.zeros((len(self.line_list)))
        i = 0
        for line in self.line_list:
            q = np.array(line[0])
            s = np.array(line[1]) - q
            dist[i] = self.Intersection(p,r,q,s)
            i += 1
        return np.min(dist)

    def Intersection(self,p,r,q,s):
        if np.cross(r, s) == 0 and np.cross((q-p), r) == 0:    # collinear
            t0 = np.dot(q-p, r)/np.dot(r, r)
            t1 = t0 + np.dot(s, r)/np.dot(r, r)
            if ((np.dot(s, r) > 0) and (0 <= t1 - t0 <= 1)) or ((np.dot(s, r) <= 0) and (0 <= t0 - t1 <= 1)):
                #print('collinear and overlapping, q_s in p_r')
                return 0.0
            else:
                #print('collinear and disjoint')
                return np.linalg.norm(r)
        elif np.cross(r, s) == 0 and np.cross((q-p), r) != 0:  # parallel r × s = 0 and (q − p) × r ≠ 0,
            #print('parallel')
            return np.linalg.norm(r)
        else:
            t = np.cross((q - p), s) / np.cross(r, s)
            u = np.cross((q - p), r) / np.cross(r, s)
            if 0 <= t <= 1 and 0 <= u <= 1:
                #print('intersection: ', p + t*r)
                return t*np.linalg.norm(r)
            else:
                #print('not parallel and not intersect')
                return np.linalg.norm(r)

def Draw(bot_pos, line_list, sensor_data, bot_param, scale=5.0, view=(512,512)):
    img = 255*np.ones((512,512,3), np.uint8)
    
    for line in line_list:
        pt1 = np.round(scale * line[0])
        pt2 = np.round(scale * line[1])
        cv2.line(img, tuple(pt1.astype(np.int).tolist()), tuple(pt2.astype(np.int).tolist()), (0,0,0), 3)
    
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        theta = bot_pos[2] + bot_param[1] + i*inter
        cv2.line(
            img, 
            (int(scale*bot_pos[0]), int(scale*bot_pos[1])), 
            (   int(scale*(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta)))), 
                int(scale*(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))))  ),
            (255,0,0), 1)
    
    cv2.circle(img,(int(scale*bot_pos[0]), int(scale*bot_pos[1])), int(scale*1.5), (0,0,255), -1)
    return img

def SensorMap(m, bot_pos, bot_param, sensor_data):
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        theta = bot_pos[2] + bot_param[1] + i*inter
        m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
        int(bot_pos[1]),
        int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
        )

def MappingProcess(env):
    sensor_data = env.Sensor()
    img = Draw(env.bot_pos, env.line_list, sensor_data, env.bot_param)
    SensorMap(m, env.bot_pos, env.bot_param, sensor_data)
    #mimg = m.GetMapProb(0, 200, 0, 200)
    mimg = m.GetMapProb(m.boundary[0]-20, m.boundary[1]+20, m.boundary[2]-20, m.boundary[3]+20)
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)

    return img, mimg

if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)

    # Initialize Grid Map
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [0.4, -0.4, 5.0, -5.0] 
    m = GridMap(map_param, gsize=0.5)

    # Initialize 2D Environment
    # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
    bot_param = [60, 30.0, 150.0, 50.0, 1.0, 3.0]
    bot_pos = np.array([40.0, 40.0, 0.0])
    env = SingleBotLaser2D(bot_pos, bot_param)
    env.line_list.append([np.array((10,10)), np.array((10,90))])
    env.line_list.append([np.array((10,10)), np.array((80,10))])
    env.line_list.append([np.array((10,90)), np.array((80,90))])
    env.line_list.append([np.array((80,10)), np.array((80,90))])

    env.line_list.append([np.array((40,10)), np.array((50,30))])
    env.line_list.append([np.array((45,10)), np.array((55,30))])
    env.line_list.append([np.array((50,30)), np.array((55,30))])

    env.line_list.append([np.array((70,50)), np.array((60,90))])
    env.line_list.append([np.array((70,50)), np.array((80,50))])
    
    env.line_list.append([np.array((20,70)), np.array((40,70))])
    env.line_list.append([np.array((20,70)), np.array((20,90))])
    env.line_list.append([np.array((40,70)), np.array((40,90))])

    # Initialize Mapping
    img, mimg = MappingProcess(env)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Main Loop
    while(1):
        # Input Control
        k = cv2.waitKey(1)
        if k==ord('w'):
            env.BotAction(1)
        if k==ord('s'):
            env.BotAction(2)
        if k==ord('a'):
            env.BotAction(3)
        if k==ord('d'):  
            env.BotAction(4)
        
        if k==ord('i'):
            env.BotAction(5)
        if k==ord('j'):
            env.BotAction(6)
        if k==ord('l'):
            env.BotAction(7)
        if k==ord('k'):
            env.BotAction(8)
        
        if k>0:
            img, mimg = MappingProcess(env)
            cv2.imshow('view',img)
            cv2.imshow('map',mimg)
        
    cv2.destroyAllWindows()