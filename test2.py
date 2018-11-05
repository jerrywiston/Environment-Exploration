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

def SensorMapping(m, bot_pos, bot_param, sensor_data):
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        theta = bot_pos[2] + bot_param[1] + i*inter
        m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta-90))),
        int(bot_pos[1]),
        int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta-90)))
        )

if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)

    # Initialize 2D Environment
    # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
    bot_param = [240,-30.0, 210.0, 150.0, 6.0, 6.0]
    bot_pos = np.array([100.0, 100.0, 90.0])
    motion = SimpleMotionModel(0.5, 0.5, 0.3)
    env = SingleBotLaser2D(bot_pos, bot_param, 'Image/map_large.png', motion)

    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [0.4, -0.4, 5.0, -5.0]
    m = GridMap(map_param, gsize=1.0)
    sensor_data = env.Sensor()
    SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)

    img = DrawEnv(env.img_map, 1, env.bot_pos, sensor_data, env.bot_param)
    mimg = AdaptiveGetMap(m)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Initialize Particle
    total_particle = 30
    pf = ParticleFilter(bot_pos.copy(), bot_param, copy.deepcopy(m), motion, total_particle)
    sensor_data_rec = sensor_data.copy()
    pimg = AdaptiveGetMap(pf.particle_list[0].gmap)
    cv2.imshow('particle_map', pimg)

    # Scan Matching Test
    matching_m = GridMap(map_param, gsize=1.0)
    SensorMapping(matching_m, env.bot_pos, env.bot_param, sensor_data)
    matching_pos = bot_pos.copy()
    path = [bot_pos.copy()]
    matching_path = [bot_pos.copy()]

    # Main Loop
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
        
        if action > 0:
            env.BotAction(action)
            sensor_data = env.Sensor()
            path.append(env.bot_pos.copy())
            SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)
    
            img = DrawEnv(env.img_map, 1, env.bot_pos, sensor_data, env.bot_param)
            mimg = AdaptiveGetMap(m)
            
            Neff = pf.Feed(action, sensor_data)
            if Neff < total_particle / 2:
                pf.Resampling(sensor_data)
            img = DrawParticle(img, pf.particle_list)
            mid = np.argmax(pf.weights)
            imgp = AdaptiveGetMap(pf.particle_list[mid].gmap)
            #particle_path = pf.particle_list[mid].trajectory
            
            pc = SensorData2PointCloud(sensor_data_rec, env.bot_pos, env.bot_param)
            xc = SensorData2PointCloud(sensor_data, env.bot_pos, env.bot_param)
            R,T = Icp2d.Icp(300,pc,xc)
            Ttot = np.array([[matching_pos[0], matching_pos[1]]])
            Ttot = Icp2d.Transform(Ttot, R, T)[0]
            theta = matching_pos[2]
            Rtot = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                             [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
            deg = Rotation2Deg(np.matmul(R,Rtot))
            matching_pos = [Ttot[0], Ttot[1], deg]
            matching_path.append(np.array(matching_pos))
            aimg = DrawAlign(xc, pc, R, T)
            
            SensorMapping(matching_m, matching_pos, env.bot_param, sensor_data)
            matching_img = AdaptiveGetMap(matching_m)
            img = DrawPath(img, matching_path, color=(200,50,50))
            for i in range(total_particle):
                particle_path = pf.particle_list[i].trajectory
                img = DrawPath(img, particle_path, color=(50,200,50))
            img = DrawPath(img, path, color=(50,50,200))

            cv2.imshow('align',aimg)
            cv2.imshow('view',img)
            cv2.imshow('map',mimg)
            cv2.imshow('matching_map',matching_img)
            cv2.imshow('particle_map',imgp)
            
            sensor_data_rec = sensor_data.copy()

    cv2.destroyAllWindows()