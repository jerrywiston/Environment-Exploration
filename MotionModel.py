import numpy as np
import matplotlib.pyplot as plt

class SimpleMotionModel:
    def __init__(self, normal_var, tangent_var, angular_var):
        self.normal_var = normal_var
        self.tangent_var = tangent_var
        self.angular_var = angular_var

    def Sample(self, pos_, n, t, theta):
        pos = pos_.copy()
        n = n + self.normal_var*np.random.randn()
        t = t + self.tangent_var*np.random.randn()
        theta = theta + self.angular_var*np.random.randn()
        pos[2] = (pos[2] + theta) % 360
        pos[0] = pos[0] + n*np.cos(np.deg2rad(pos[2])) + t*np.sin(np.deg2rad(pos[2]))
        pos[1] = pos[1] + n*np.sin(np.deg2rad(pos[2])) + t*np.cos(np.deg2rad(pos[2]))
        return pos

if __name__ == '__main__':
    m_model = SimpleMotionModel(0.1, 0.1, 3)
    a = np.zeros((100,3))
    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "b.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "g.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "r.")

    plt.axis('equal')
    plt.show()
