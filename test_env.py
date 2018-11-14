import numpy as np
import cv2
from SingleBotLaser2Dgrid import *
from GridMap import *
from ParticleFilter import *
import utils
import copy
from Viewer import *
import Icp2d
from MotionModel import *

if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('env', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('obs', cv2.WINDOW_AUTOSIZE)

    # Initialize 2D Environment
    # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
    bot_param = [120,-30.0, 210.0, 150.0, 6.0, 6.0]
    bot_pos = np.array([100.0, 50.0, -90.0])
    motion = SimpleMotionModel(0.2, 0.2, 0.1)
    env = SingleBotLaser2D(bot_pos, bot_param, 'Image/map3.png', motion)

    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [-0.4, 0.3, 5.0, -5.0]
    m = GridMap(map_param, gsize=2.0)
    sensor_data = env.Sensor()
    SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)

    img = DrawEnv(env.img_map)
    img = DrawBot(img, env.bot_pos, sensor_data, env.bot_param)
    mimg = AdaptiveGetMap(m)
    oimg = m.getObs(bot_pos,50,50)
    cv2.imshow('obs',oimg)
    cv2.imshow('env',img)
    cv2.imshow('map',mimg)

    # Main Loop
    step = 0
    while(1):
        # Input Control
        action = -1
 
        k = cv2.waitKey(1)
        if k==ord('w'):
            action = 1
        if k==ord('s'):
            action = 2
        if k==ord('a'):
            action = 3
        if k==ord('d'): 
            action = 4 

        action = np.random.randint(4) + 1
        
        if action > 0:
            collision = env.BotAction(action)
            if collision:
                print(step, "Bang !!")
            else:
                print(step, "Safe~")
            sensor_data = env.Sensor()
            info_gain = SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)
            print("Info Gain:", info_gain)
    
            img = DrawEnv(env.img_map)
            img = DrawPath(img, env.path)
            img = DrawBot(img, env.bot_pos, sensor_data, env.bot_param)
            mimg = AdaptiveGetMap(m)
            oimg = m.getObs(env.bot_pos,50,50)
            print(oimg.shape)
            cv2.imshow('obs',oimg)
            cv2.imshow('env',img)
            cv2.imshow('map',mimg)
            step += 1

    cv2.destroyAllWindows()