import os
import time
import copy
import random
import joblib
import sklearn
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial
import sys
sys.path.append("..")
#LIBRARY OF PYTHON
import tensorflow as tf
#LIBRARY OF TENSORFLOW
from models.model import Model
from utils.dataloader import DataLoad
from utils.get_tensor import *
from utils.visualize import *
#SELF DEFINED OBJECT
#---------------------------------------------------------------
DATASET_INDEX = 0
META_NAME = 'model0'
CKPT_NAME = 'ckpt0'
DATA_DIR = '/home/zt/zt_datas/'
DATA_FILE = 'processed_data.csv'
RELOAD_MODLE_DIR = 'saved_models/'
RELOAD_MODEL = 'best_model.ckpt-135.meta'
BEST_MODLE_DIR = './train0515/LSTM_MDN/LSTM_ckpt0516_0/'
BEST_MODEL = 'best_model.ckpt-163.meta'
#Direction of Training SET, Testing SET 
TRAIN_X_DIR   = 'TRAIN_TEST_DATA/train_x.pkl'
TRAIN_Y_DIR   = 'TRAIN_TEST_DATA/train_y.pkl'
TEST_X_DIR    = 'TRAIN_TEST_DATA/test_x.pkl'
TEST_Y_DIR    = 'TRAIN_TEST_DATA/test_y.pkl'
EXAMPLE_X_DIR = 'example_data/example_x'+str(DATASET_INDEX)+'.pkl'
EXAMPLE_Y_DIR = 'example_data/example_y'+str(DATASET_INDEX)+'.pkl'
#---------------------------------------------------------------
def load_arg():
    paser = argparse.ArgumentParser()
    paser.add_argument("--train_eval", action="store_true",
                      default=False, help="train or eval")
    paser.add_argument("--train_continous", action="store_true",
                      default=False, help="continous train or not")
    paser.add_argument("--get_new_data", action="store_true",
                      default=False, help="split the train test data")
    paser.add_argument("--save_mdn_params", type=bool,
                      default=False, help="save mdn params or not")
    paser.add_argument("--visualize", type=bool,
                      default=True, help="plot the results")
    paser.add_argument("--use_attention", type=bool, default=True,
                      help="use attention")
    paser.add_argument("--attn_input_feeding", type=bool, default=False,
                      help="attention")
    paser.add_argument("--use_regularization", type=bool, default=True,
                      help="use regularization")
    paser.add_argument("--use_mdn", type=bool, default=True,
                      help="use mdn") 
    paser.add_argument("--use_beam_search", type=bool, default=False,
                      help="use beam search")
    paser.add_argument('--obj_num', type=int, default=7,
                      help="number of object")
    paser.add_argument('--components', type=int, default=6,
                      help="number of gaussian mixtures")
    paser.add_argument("--hidden_layers", type=int,
                      default=2, help="number of hidden layer ")
    paser.add_argument("--encoder_len", type=int, default=50,
                      help="encoder length")
    paser.add_argument("--decoder_len", type=int, default=30,
                      help="decoder length")
    paser.add_argument("--attention_type", type=str, default='luong',
                      help="attention type")
    paser.add_argument("--hidden_size", type=int, default=128,
                      help="units num in each hidden layer")
    paser.add_argument("--drop_out", type=float, default=0.7,
                      help="drop out probability")
    paser.add_argument('--learning_rate', type=float, default=0.002,
                      help="learning_rate")
    paser.add_argument('--epoch', type=int, default=300,
                      help="epoch")
    paser.add_argument('--batch_size', type=int, default=16,
                      help="batch size")
    paser.add_argument('--input_size', type=int, default=17,
                      help="input size, feature size")
    paser.add_argument('--model_type', type=str, default='LSTM_MDN',
                      help='the model type should be LSTM, \
                        BLSTM, LSTM_MDN or BLSTM_MDN.')
    args = paser.parse_args()
    return args
   
def main():
    #step 1: GET ARGS FOR MODEL -------------------------------
    args = load_arg()
    TrainIndicator     = args.train_eval
    TrainContinous     = args.train_continous
    GetDataIndicator   = args.get_new_data
    MdnParamsIndicator = args.save_mdn_params
    #step 2: GET TRAIN/TEST DATA-------------------------------
    if GetDataIndicator:
        direc = DATA_DIR
        csv_file = DATA_FILE
        dl = DataLoad(direc, csv_file)
        dl.read_data()
        sum_samples, num_train, num_test = dl.test_valid_data_split(ratio=0.8)
        X_train = dl.data['X_train']
        y_train = dl.data['y_train']
        X_test = dl.data['X_test']
        y_test = dl.data['y_test']
    else:
        X_train = joblib.load(TRAIN_X_DIR)
        y_train = joblib.load(TRAIN_Y_DIR)
        # X_test = joblib.load(TEST_X_DIR)
        # y_test = joblib.load(TEST_Y_DIR) 
        X_test = joblib.load(EXAMPLE_X_DIR)
        y_test = joblib.load(EXAMPLE_Y_DIR) 
        num_train = len(X_train)
        num_test = len(X_test)
    train_batch_num = int(num_train/args.batch_size)
    test_batch_num = int(num_test/args.batch_size)
    num_test_ater_training = int(num_test)
    print ('TRAINING LENGTH = ', num_train)
    print ('TESTING LENGTH = ', num_test)
    print ('TEST AFTER TRAINING = ', num_test_ater_training)
    #step 3: TRAIN: CONSTRUCT THE MODEL ----------------------
    if TrainIndicator:
        print ('TRAINING........................................')
        tf.reset_default_graph()
        model = Model(args)
        if args.model_type == 'LSTM':
            print ('LSTM MODEL.')
            model.LSTM_model()
        elif args.model_type == 'BLSTM':
            print ('BLSTM MODEL.')
            model.bidir_LSTM_model()
        elif args.model_type == 'LSTM_MDN':
            print ('LSTM MDN MODEL.')
            model.MDN_model('LSTM')
        elif args.model_type == 'BLSTM_MDN':
            print ('BLSTM MDN MODEL')
            model.MDN_model('BLSTM')
        else:
            print ("please choose correct model type")
            return
        model.Evaluating()
        #step 4: START TRAINING -----------------------------
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('log/'+ args.model_type+ '/',sess.graph)
            start_time = time.time()
            train_cost_list = []#COST OF EACH EPOCH
            test_cost_list = []#COST OF EACH EPOCH
            y_pred_list = []
            train_cost_reference = 1000.0
            best_test_cost = 1000.0
            if TrainContinous:#RELOAD MODEL, CONTINOUS TRAINING
                model_saver = tf.train.import_meta_graph(RELOAD_MODLE_DIR+ RELOAD_MODEL)
                model_saver.restore(sess,tf.train.latest_checkpoint(RELOAD_MODLE_DIR))    
                graph = tf.get_default_graph()
                inputx = graph.get_tensor_by_name("input_data:0")
                inputy = graph.get_tensor_by_name("ground_truth:0")
                dropout = graph.get_tensor_by_name("drop_out:0")
                trainop = graph.get_tensor_by_name("evaluating/Adam:0")
                costop = graph.get_tensor_by_name("evaluating/add_31:0")
            for i in range(args.epoch):
                train_cost_batch_list = []
                for batch_num in range(train_batch_num): 
                    rand_ind = np.random.choice(num_train, args.batch_size, replace=False)
                    if TrainContinous:
                        feed_dict = {inputx: X_train[rand_ind], inputy: y_train[rand_ind], dropout: args.drop_out}
                        fetch = [trainop, costop]
                        _, train_cost = sess.run([trainop, costop], feed_dict=feed_dict)  
     
                    else:
                        feed_dict = {model.X: X_train[rand_ind], model.y: y_train[rand_ind], model.drop_out: args.drop_out}
                        fetch = [model.train_op, model.cost, model.global_step, model.merged_summary_op]
                        _, train_cost, step, summary = sess.run(fetch, feed_dict=feed_dict)  
                        train_writer.add_summary(summary, batch_num)
                        #print ('train cost = ', train_cost, 'mdn cost = ', mdn_cost, 'reg2 cost = ', reg2_cost)
                        if batch_num == train_batch_num - 1:
                            train_writer.close()
                    train_cost_batch_list.append(train_cost)
                train_cost_mean = np.mean(train_cost_batch_list)
                train_cost_list.append(train_cost_mean)
                test_cost_batch_list = []
                # SHUFFLE TEST DATA
                y_pred_batch_list = []
                X_test, y_test = shuffle(X_test, y_test, random_state=i * 42)
                for start, end in zip(range(0, num_test, args.batch_size),range(args.batch_size, num_test + 1, args.batch_size)):
                    feed_dict = {model.X: X_test[start:end], model.y: y_test[start:end], model.drop_out: 1.0}
                    ground_truth = copy.deepcopy(y_test[start:end])      
                    fetch = [model.cost, model.decoder_outputs, model.numel]
                    test_cost_batch, y_pred, numel = sess.run(fetch, feed_dict=feed_dict)
                    test_cost_batch_list.append(test_cost_batch)
                    #ground_truth=ground_truth.tolist()
                    y_pred = np.reshape(np.array(y_pred), (-1, 2*args.obj_num))
                    ground_truth = ground_truth.reshape((-1, 2*args.obj_num))
                    y_pred_batch_list.append(zip(y_pred,ground_truth))
                # print ("ground_truth shape{}".format(ground_truth.shape))
                # print ("y_pred shape{}".format(y_pred.shape))
                test_cost_mean = np.mean(test_cost_batch_list)
                if test_cost_mean < best_test_cost:
                    best_test_cost = test_cost_mean
                    model.saver.save(sess, 'TRAINED_MODEL/'+ args.model_type + '/best_model.ckpt', global_step=i+1)
                train_cost = np.mean(train_cost_list)
                test_cost_list.append(test_cost_mean)
                y_pred_list.append(y_pred_batch_list)
                print ("at %d epoch, the training cost is %f"%(i, train_cost_mean))
                print ("at %d epoch, the test cost is %f"%(i, test_cost_mean))
                #print ("at {} epoch, the test_AUC is {}".format(i, test_AUC))
                print ("------------------------------------------------------")
    
            best_cost = min(test_cost_list) 
            best_cost_ind = test_cost_list.index(best_cost) 
            best_y_pred = np.squeeze(y_pred_list[best_cost_ind])
            end_time = time.time()
            spend_time = end_time - start_time
            print ("========================================================")
            print ("Finally, the model has {} parameters\n\n".format(numel))
          # wirte result in local
            with open(args.model_type + '.txt', 'a') as f:
                f.write("the best test cost is {} at {} epoch, the model has {} parameters, lr_rate is {}, dropout is {}, batchsize is {}, spend time is {}, \n\n"
                    .format(best_cost, best_cost_ind, numel, args.learning_rate, args.drop_out, args.batch_size, spend_time))

    #step 3: TEST: RELOAD THE MODEL ----------------------
    if not TrainIndicator:
        print ('TESTING.........................................')
        past_location_everybatch = []#GROUND TRUTH
        y_pred_everybatch = []#PREDICTION (x, y)
        params_of_7car   = [[],[],[],[],[],[],[]]#PREDICTED GAUSSIAN PARAMS (mux, muy, sigmax, sigmay)*COMPONENTS*OBJ_NUM
        rho_of_7car      = [[],[],[],[],[],[],[]]#PREDICTED GAUSSIAN PARAMS (rho)*COOMPONENTS*OBJ_NUM
        subprob_of_7car  = [[],[],[],[],[],[],[]]#PREDICTED GAUSSIAN PARAMS (prob_m)*COOMPONENTS*OBJ_NUM
        prob_of_7car     = [[],[],[],[],[],[],[]]#PREDICTED GAUSSIAN PARAMS (prob)*OBJ_NUM
        pi_of_7car       = [[],[],[],[],[],[],[]]#PREDICTED GAUSSIAN PARAMS (pi)*COOMPONENTS*OBJ_NUM
   
        tf.reset_default_graph()
        restore_graph = tf.Graph()
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name,'\n')
        with tf.Session(graph=restore_graph) as restore_sess:
            graph_dir = 'MDN_over_next_vector/'
            tensor_getter = TensorGetter(restore_graph, graph_dir, restore_sess, args.obj_num, args.components, args.batch_size)  
            # restore_saver = tf.train.import_meta_graph('TRAINED_MODEL/'+args.model_type+'/'+str(DATE)+METAL_NAME)
            # restore_saver.restore(restore_sess,tf.train.latest_checkpoint('TRAINED_MODEL/'+args.model_type+'/'+str(DATE)+CKPT_NAME)) 
            #LSTM_MDN MODEL -----------------------------------------------------------------------------------
            restore_saver = tf.train.import_meta_graph(BEST_MODLE_DIR+ BEST_MODEL)
            restore_saver.restore(restore_sess,tf.train.latest_checkpoint(BEST_MODLE_DIR))
            #LSTM MODEL ---------------------------------------------------------------------------------------
            # restore_saver = tf.train.import_meta_graph('./train0515/LSTM/LSTM_ckpt0516/best_model.ckpt-180.meta')
            # restore_saver.restore(restore_sess,tf.train.latest_checkpoint('./train0515/LSTM/LSTM_ckpt0516/'))          
            for start, end in zip(range(0, num_test_ater_training, args.batch_size),range(args.batch_size, num_test_ater_training + 1, args.batch_size)):
                inputx = restore_graph.get_tensor_by_name("input_data:0")
                inputy = restore_graph.get_tensor_by_name("ground_truth:0")
                dropout = restore_graph.get_tensor_by_name("drop_out:0")
                feed_dict = {inputx: X_test[start:end], inputy: y_test[start:end], dropout:1.0}
                past_location = copy.deepcopy(X_test[start:end])
                ground_truth = copy.deepcopy(y_test[start:end])
                print ("feed finished")
                final_output = []
          
                for step in range(args.decoder_len):
                    if step == 0:
                        op_name = str('rnn_decoder/add:0')
                    else:
                        op_name = str('rnn_decoder/add_')+str(step)+str(':0')
                    output = restore_graph.get_tensor_by_name(op_name)              
                    restore_sess.run(output,feed_dict)      
                    output_step = output.eval(feed_dict= feed_dict, session = restore_sess)                  
                    final_output.append(output_step)         
                final_output_reshape = np.stack(final_output,axis=1)
                final_result = np.concatenate((ground_truth,final_output_reshape),axis=2)
                print ("final_result shape is:{}".format(final_result.shape))#[batch_size, decoder_len, 2*OBJ_NUM]
                y_pred_everybatch.append(final_result)
                past_location_everybatch.append(past_location)

                if MdnParamsIndicator:#SAVE MDN PARAMS OR NOT
                    #MU, SIGMA, SUBPROB, PROB, PI, PRO
                    muxTensorList_a, muyTensorList_a = tensor_getter.run_tensor(feed_dict, 'MU')
                    sigmaxTensorList_a, sigmayTensorList_a = tensor_getter.run_tensor(feed_dict, 'SIGMA')
                    subprobTensorList_a = tensor_getter.run_tensor(feed_dict, 'SUBPROB')
                    piTensorList_a = tensor_getter.run_tensor(feed_dict, 'PI')
                    rhoTensorList_a = tensor_getter.run_tensor(feed_dict, 'RHO')
                    probTensorList_a = tensor_getter.run_tensor(feed_dict, 'PROB')
                    for obj_num in range(args.obj_num):
                        result_each_car = np.reshape(np.concatenate((muxTensorList_a[obj_num], muyTensorList_a[obj_num], \
                          sigmaxTensorList_a[obj_num], sigmayTensorList_a[obj_num]), axis = 2), (-1, 4*args.components))
                        params_of_7car[obj_num].append(result_each_car)#[batch_size*decoder_len, 4*COMPONENTS]
                        rho_of_7car[obj_num].append(rhoTensorList_a[obj_num])#[batch_size*decoder_len, COMPONENTS]
                        subprob_of_7car[obj_num].append(subprobTensorList_a[obj_num])#[batch_size*decoder_len, COMPONENTS]
                        prob_of_7car[obj_num].append(probTensorList_a[obj_num])#[batch_size*decoder_len, 1]
                        pi_of_7car[obj_num].append(piTensorList_a[obj_num])#[batch_size*decoder_len, COMPONENTS]
         
            total_test_num = len(y_pred_everybatch)#NUM OF BATCHES
            print ("TESTED {} BATCHES".format(len(y_pred_everybatch)))
            for test_num in range(total_test_num):
                pred_ground_label = []
                pred_label, ground_label = [], []
                mean_std_label = []
                result_of_7car = np.zeros((1))
                for obj_num in range(args.obj_num):
                    label_p = ['truth_x'+ str(obj_num), 'truth_vy'+ str(obj_num)]
                    label_g = ['predict_x'+ str(obj_num), 'predict_vy'+ str(obj_num)]
                    pred_label.extend(label_p)
                    ground_label.extend(label_g)
                    if MdnParamsIndicator:
                        result_obj = np.concatenate((params_of_7car[obj_num][test_num], prob_of_7car[obj_num][test_num],\
                        subprob_of_7car[obj_num][test_num], rho_of_7car[obj_num][test_num], pi_of_7car[obj_num][test_num]),axis=1)
                        if obj_num == 0:
                            result_of_7car = result_obj
                        else:
                            result_of_7car = np.concatenate((result_of_7car, result_obj), axis =1)
                        meanxcom, meanycom, sigmaxcom, sigmaycom, probcom, rhocom, picom = [],[],[],[],[],[],[]
                        for com in range(args.components):
                            meanxcom.append('meanx'+ str(obj_num)+ str(com))
                            meanycom.append('meany'+ str(obj_num)+ str(com))
                            sigmaxcom.append('sigmax'+ str(obj_num)+ str(com))
                            sigmaycom.append('sigmay'+ str(obj_num)+ str(com))
                            probcom.append('prob'+ str(obj_num)+ str(com))
                            rhocom.append('rho'+ str(obj_num)+ str(com))
                            picom.append('pi'+ str(obj_num)+ str(com))
                        mean_std_label = mean_std_label + meanxcom + meanycom + sigmaxcom + sigmaycom + ['prob_total'+str(obj_num)] +\
                                        probcom + rhocom + picom
                pred_ground_label = pred_label + ground_label
                past_location_batch = np.reshape(past_location_everybatch[test_num], (-1, args.input_size))
                past_lacation_frame = pd.DataFrame(past_location_batch, columns = ['Lane_ID','Space_Hdwy','Local_Y','Local_X','v_Vel',\
                'left_xp','left_vp','xcordp','vcordp','right_xp','right_vp','left_xf','left_vf','xcordf','vcordf','right_xf','right_vf'])
                past_lacation_frame.to_csv('results/'+ args.model_type+ '/past_location_'+ str(test_num)+ '.csv', mode='w', index = None)
                y_pred_and_ground = np.reshape(y_pred_everybatch[test_num], (-1, 4*args.obj_num))
                pred_ground_frame = pd.DataFrame(y_pred_and_ground, columns = pred_ground_label)
                pred_ground_frame.to_csv('results/'+ args.model_type + '/pred_ground_'+ str(test_num)+ '.csv', mode='w', index = None)
                if MdnParamsIndicator:
                    mean_stdframe = pd.DataFrame(result_of_7car, columns = mean_std_label)
                    mean_stdframe.to_csv('results/'+ args.model_type + '/mean_str_'+ str(test_num)+ '.csv', mode='w', index = None)
                if args.visualize:
                    visualizer = Visualize(args)
                    visualizer.visualize(pred_ground_frame, 0, True)
    return 
  
if __name__ == '__main__':
    main()
    
# =========use library "hyperopt" to finetune the hyerparameters==============
'''
from hyperopt import fmin, tpe, hp, partial

batch_list = [16,32,64]

space = {"lr_rate": hp.uniform("lr_rate", 0.002, 0.004),
         "dp_out": hp.uniform("dp_out", 0.5, 1.0),
         "bt_size": hp.choice("bt_size", batch_list)}

try:
    best = fmin(main, space, algo=tpe.suggest, max_evals=50)
    best["bt_size"] = batch_list[best["bt_size"]]   
    best_cost = main(best_cost)
    with open('finetune.txt', 'a') as f:
        f.write("the best cost is {}, its lr_rate is {}, drop_out is {}, batch_size is {}\n\n".
              format( best_cost, best["lr_rate"], best["dp_out"], best["bt_size"]))
except Exception as err:
    with open("error_info.txt","a") as f:
        f.write(str(err)+'\n')
    print (err)
'''
