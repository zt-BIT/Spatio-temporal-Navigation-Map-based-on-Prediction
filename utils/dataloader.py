import os
import joblib
import tarfile
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.preprocessing import StandardScaler#To normalize the data
# get the longest elements and its index
TOTAL_FRAME = 80
PAST_FRAME = 50
FUT_FRAME = 30
DELTA_FRAME = 5
FEA_NUM = 17
LABEL_NUM = 14
 
def groupByVehicle(df):
    vehicleindex=[]
    frameindex=[]
    groups = list(df.groupby('Vehicle_ID'))
    for i in range(len(groups)):
        vehicleid = groups[i][0]
        frames = list(groups[i][1].index)
        startindex = frames[0]
        endindex = frames[-1]
        vehicleindex.append(vehicleid)
        frameindex.append([startindex, endindex])
    return vehicleindex,frameindex

def write_to_txt(file_name, content):#content is a list
    """
    file_name: str, name of the file
    content:   dict
    """
    with open(file_name, 'w') as txt_file:
        for key, value in content.items():
            txt_file.write(key)
            txt_file.write('= ')
            txt_file.write(str(value))
            txt_file.write('\n')

    

class DataLoad():
    def __init__(self, direc, csv_file):
        """Create a dataload class to load data from local and preprocess with it
        args:
            dirc: [str] the path of input data file
            csv_file: [str] the input data file name, and the file extentions should be '.csv'
        """
        assert direc[-1] == '/', 'Please provide a dicrectionary ending with a /'
        assert csv_file[-3:] == 'csv', 'Please confirm the file extentions'
        self.csv_loc = direc + csv_file  # The location of the csv file
        self.save_dir = direc + 'normalization_coef.csv'
        self.trajectory_data = []  # create a list to store the preprocessed data,each element of the list represent a seq of 8s(50) of a certain Vehicle_ID
        self.labels = []  # create a list to store the preprocessed labels actually,the labels should be in length of 30(3s) and if using the OGM/maneuver,the labels should be tagged artificially.
        self.data = {}  # create a dict to store splitted data and labels
        self.N = 0  # total number of sequences in preprocessed data
        self.iter_train = 0  # train iteration
        self.epochs = 0  # epochs for looping
        self.omit = 0  # omitted sequences number
        self.normal_dict = {}
        self.frameindex=[]
        self.vehicleindex=[]
        if not os.path.exists(self.csv_loc):
            print ('WRONG DIRECTORY')


    def read_data(self):
        #========step 1: read data=============
        #-----judgement for initial configuration--------
        df = pd.read_csv(self.csv_loc)
        df_arr = df[['Lane_ID','Space_Hdwy','Local_Y','Local_X','v_Vel','left_xp','left_vp','xcordp','vcordp','right_xp','right_vp','left_xf','left_vf','xcordf','vcordf','right_xf','right_vf']].values
        df_arr = df_arr[:100000,:]
        df = df[:100000]
        #========step 2: extract useful data==========
        row, col = df_arr.shape
        print ("row={},col={}".format(row,col))
        self.vehicleindex, self.frameindex = groupByVehicle(df)

        for i in range(len(self.vehicleindex)-1):
            totalframe = self.frameindex[i][1]-self.frameindex[i][0]
            if totalframe < TOTAL_FRAME:
                continue
            seqnum = int((totalframe - TOTAL_FRAME)/DELTA_FRAME)
            for num in range(0,seqnum,1): 
                start_index = self.frameindex[i][0]+ DELTA_FRAME*num
                end_index = start_index + TOTAL_FRAME
                seq = df_arr[start_index:end_index,:]
                self.trajectory_data.append(seq[:PAST_FRAME, :])
                self.labels.append(seq[PAST_FRAME:, 3:])

        print ("trajectory_data.shape=%d"%(len(self.trajectory_data)))
        print ("label.shape=%d"%(len(self.labels)))
        try:
            self.trajectory_data = np.stack(self.trajectory_data, 0)#to make the element of the list to form a matrix
            self.labels = np.stack(self.labels, 0)
            self.N = len(self.labels)
        except:
            print ('Something is wrong when convert list to ndarray')
        
 

    def test_valid_data_split(self, ratio=0.8):
        """split test and vlid data"""
        per_ind = np.random.permutation(self.N)  # shuffle the index
        # set the top ratio indexes as test index
        train_ind = per_ind[:int(ratio * self.N)]
        # set the left indexes as test index
        test_ind = per_ind[int(ratio * self.N):]
        self.data['X_train'] = self.trajectory_data[train_ind]
        self.data['y_train'] = self.labels[train_ind]
        self.data['X_test']  = self.trajectory_data[test_ind]
        self.data['y_test']  = self.labels[test_ind]
        print ("self.data['X_train'] shape:{}".format(self.data['X_train'].shape))
        print ("self.data['y_train'] shape:{}".format(self.data['y_train'].shape))
        print ("self.data['X_test'] shape:{}".format(self.data['X_test'].shape))
        print ("self.data['y_test'] shape:{}".format(self.data['y_test'].shape))
        xtrainvalues = self.data['X_train'].reshape((-1, FEA_NUM))
        xtestvalues  = self.data['X_test'].reshape((-1, FEA_NUM))
        ytrainvalues = self.data['y_train'].reshape((-1, LABEL_NUM))
        ytestvalues  = self.data['y_test'].reshape((-1, LABEL_NUM))
       
        self.scaler = StandardScaler().fit(xtrainvalues)
        xtrainvalues = self.scaler.transform(xtrainvalues)
        self.data['X_train'] = xtrainvalues.reshape((-1, PAST_FRAME, FEA_NUM))
        self.normal_dict['x_train_mean'] = self.scaler.mean_
        self.normal_dict['x_train_var']  = self.scaler.var_
        self.scaler = StandardScaler().fit(xtestvalues)
        xtestvalues = self.scaler.transform(xtestvalues)
        self.data['X_test']=xtestvalues.reshape((-1, PAST_FRAME, FEA_NUM))
        self.normal_dict['x_test_mean'] = self.scaler.mean_
        self.normal_dict['x_test_var']  = self.scaler.var_
        self.scaler = StandardScaler().fit(ytrainvalues)
        ytrainvalues = self.scaler.transform(ytrainvalues)
        self.data['y_train']=ytrainvalues.reshape((-1, FUT_FRAME, LABEL_NUM))
        self.normal_dict['y_train_mean'] = self.scaler.mean_
        self.normal_dict['y_train_var']  = self.scaler.var_
        self.scaler = StandardScaler().fit(ytestvalues)
        ytestvalues = self.scaler.transform(ytestvalues)
        self.data['y_test']=ytestvalues.reshape((-1, FUT_FRAME, LABEL_NUM))
        self.normal_dict['y_test_mean'] = self.scaler.mean_
        self.normal_dict['x_test_var']  = self.scaler.var_
        joblib.dump(self.normal_dict, self.save_dir)
        if np.any(np.isnan(self.data['X_train'] )):
            print (np.where(np.isnan(self.data['X_train'])))
            print ("X_train is nan")
        if np.any(np.isnan(self.data['y_train'] )):
            print (np.where(np.isnan(self.data['y_train'])))
            print ("y_train is nan")
        if np.any(np.isnan(self.data['X_test'] )):
            print (np.where(np.isnan(self.data['X_test'])))
            print ("X_test is nan")
        if np.any(np.isnan(self.data['y_test'] )):
            print (np.where(np.isnan(self.data['y_test'])))
            print ("y_test is nan")
        num_train = len(self.data['X_train'])
        num_test = len(self.data['X_test'])
        sum_num = num_test + num_train

        write_to_txt('normalization_coef.txt', self.normal_dict)
        print ("{0} samples in sum, including {1} traing samples, and {2} test samples".format(sum_num, num_train, num_test))
    
        return sum_num, num_train, num_test

