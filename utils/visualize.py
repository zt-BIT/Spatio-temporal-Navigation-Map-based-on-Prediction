import copy
import math
import time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

y_test_mean =  [11.77618828,33.2799103 ,28.44695125,44.78821774,45.74728455,48.5733414,\
                12.1373622,32.69251613,29.50079059,44.3767922,47.1674697,48.85264786]
y_test_var = [128.38297903,494.78487455,299.60689271,210.48545619,298.47581048,245.6518824,\
              122.94039372,413.37922631,279.26823473,124.02350834,231.40672044,160.66085139]

class DataName:
    tx = 2
    ty = 3
    px = 16
    py = 17


class Visualize(object):
    def __init__(self, args):  
        self.true_x = [[], [], [], [], [], []]#OBJ_NUM = 6
        self.true_y = [[], [], [], [], [], []]#OBJ_NUM = 6
        self.pred_x = [[], [], [], [], [], []]#OBJ_NUM = 6
        self.pred_y = [[], [], [], [], [], []]#OBJ_NUM = 6
        self.obj_num = args.obj_num - 1
        self.output_size = args.decoder_len
        self.bundle_num = 5
        self.fig = plt.figure(23,dpi=128,figsize=(10,6))
        self.timeseq = range(0, self.output_size)
        self.model = args.model_type
    def visualize(self, dataframe, plot_index, show = False):
        """
        dataframe: DataFrame
        """
        data = dataframe.values
        for obj in range(self.obj_num):
            self.true_x[obj] = data[:, DataName.tx+obj*2].tolist()
            self.true_y[obj] = data[:, DataName.ty+obj*2].tolist()
            self.pred_x[obj] = data[:, DataName.px+obj*2].tolist()
            self.pred_y[obj] = data[:, DataName.py+obj*2].tolist()
        self.normalize()
        for figr in range(1, self.obj_num+1):
            ax = self.fig.add_subplot(2, 3, figr, projection ='3d')
            ax.set_title('Car'+ str(figr), fontdict={'fontsize': 8, 'fontweight': 'medium'})
            ax.set_xlabel('x(ft)', fontsize=8)
            ax.set_ylabel('vy(ft/s)', fontsize=8)
            ax.set_zlabel('t(0.1s)', fontsize=8)
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.tick_params(axis='both', which='major', labelsize=6)
            for i in range(plot_index, plot_index + self.bundle_num):
                ax.scatter(self.true_x[figr-1][self.output_size*i:self.output_size*(i+1)],\
                           self.true_y[figr-1][self.output_size*i:self.output_size*(i+1)],self.timeseq[:], c='r', linewidths = 0.01)
                ax.scatter(self.pred_x[figr-1][self.output_size*i:self.output_size*(i+1)],\
                           self.pred_y[figr-1][self.output_size*i:self.output_size*(i+1)],self.timeseq[:], c='b', linewidths = 0.01)
        self.fig.suptitle(self.model, fontsize=10)
        if show:
            plt.show()

    def normalize(self):
        y_test_std = np.sqrt(y_test_var).tolist()
        for obj in range(self.obj_num):#6
            meanx = y_test_mean[obj*2]
            stdx  = y_test_std[obj*2]
            meany = y_test_mean[obj*2+1]
            stdy  = y_test_std[obj*2+1]
            self.true_x[obj] = [c*stdx + meanx for c in self.true_x[obj]]
            self.pred_x[obj] = [c*stdx + meanx for c in self.pred_x[obj]]
            self.true_y[obj] = [c*stdy + meany for c in self.true_y[obj]]
            self.pred_y[obj] = [c*stdy + meany for c in self.pred_y[obj]]
           















