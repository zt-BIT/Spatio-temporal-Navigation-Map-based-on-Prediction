import numpy as np
class TensorGetter(object):
    def __init__(self, graph, graph_dir, sess, obj_num, component_num, batch_size):  
        self.graph = graph
        self.dir = graph_dir
        self.sess = sess
        self.mux_tensor_name = []
        self.muy_tensor_name = []
        self.sigmax_tensor_name = []
        self.sigmay_tensor_name = []
        self.subprob_tensor_name = []
        self.prob_tensor_name = []
        self.pi_tensor_name = []
        self.rho_tensor_name = []
        self.zero_3d = np.zeros([batch_size, component_num, 1])
        self.zero_2d = np.zeros([batch_size, 1])
        self.obj_num = obj_num
        self.component_num = component_num
    def get_tensor(self, tensor_name):
        if tensor_name == 'MU':
            mux_tensor = []
            muy_tensor = []
            for i in range(self.obj_num):
                x_tensor_name = self.dir + 'Sub_' + str(1+3*i) + ':0'
                y_tensor_name = self.dir + 'Sub_' + str(2+3*i) + ':0'
                x_tensor = self.graph.get_tensor_by_name(x_tensor_name)
                y_tensor = self.graph.get_tensor_by_name(y_tensor_name)
                mux_tensor.append(x_tensor)
                muy_tensor.append(y_tensor)
            self.mux_tensor_name = mux_tensor
            self.muy_tensor_name = muy_tensor
        if tensor_name == 'SIGMA':
            sigmax_tensor = []
            sigmay_tensor = []
            for i in range(self.obj_num):
                x_tensor_name = self.dir + 'Exp_' + str(1+4*i) + ':0'
                y_tensor_name = self.dir + 'Exp_' + str(2+4*i) + ':0'
                x_tensor = self.graph.get_tensor_by_name(x_tensor_name)
                y_tensor = self.graph.get_tensor_by_name(y_tensor_name)
                sigmax_tensor.append(x_tensor)
                sigmay_tensor.append(y_tensor)
            self.sigmax_tensor_name = sigmax_tensor
            self.sigmay_tensor_name = sigmay_tensor
        if tensor_name == 'SUBPROB':
            subprob_tensor = []
            for i in range(self.obj_num):
                subprob_tensor_name = self.dir + 'div_' + str(4+5*i) + ':0'
                pr_tensor = self.graph.get_tensor_by_name(subprob_tensor_name)
                subprob_tensor.append(pr_tensor)
            self.subprob_tensor_name = subprob_tensor
        if tensor_name == 'PROB':
            prob_tensor = []
            for i in range(self.obj_num):
                prob_tensor_name = self.dir + 'Sum_' + str(1+2*i) + ':0'
                p_tensor = self.graph.get_tensor_by_name(prob_tensor_name)
                prob_tensor.append(p_tensor)
            self.prob_tensor_name = prob_tensor
        if tensor_name == 'PI':
            pi_tensor = []
            for i in range(self.obj_num):
                if i == 0:
                    pi_tensor_name = self.dir + 'Mul:0'
                else:
                    pi_tensor_name = self.dir + 'Mul_' + str(6*i) + ':0'
                pi = self.graph.get_tensor_by_name(pi_tensor_name)
                pi_tensor.append(pi)
            self.pi_tensor_name = pi_tensor
        if tensor_name == 'RHO':
            rho_tensor = []
            for i in range(self.obj_num):
                if i == 0:
                    rho_tensor_name = self.dir + 'Tanh:0'
                else:
                    rho_tensor_name = self.dir + 'Tanh_' + str(i) + ':0'
                rho = self.graph.get_tensor_by_name(rho_tensor_name)
                rho_tensor.append(rho)
            self.rho_tensor_name = rho_tensor
    def run_tensor(self, feed_data, tensor_name):
        self.get_tensor(tensor_name)
        if tensor_name == 'MU':
            mux_tensor  = self.sess.run(self.mux_tensor_name,feed_dict=feed_data)
            muy_tensor  = self.sess.run(self.muy_tensor_name,feed_dict=feed_data)
            mux_tensor  = [np.transpose(np.concatenate((mux, self.zero_3d), axis=2), [0,2,1]) for mux in mux_tensor]
            muy_tensor  = [np.transpose(np.concatenate((muy, self.zero_3d), axis=2), [0,2,1]) for muy in muy_tensor]
            return mux_tensor, muy_tensor
        if tensor_name == 'SIGMA':
            sigmax_tensor  = self.sess.run(self.sigmax_tensor_name,feed_dict=feed_data)
            sigmay_tensor  = self.sess.run(self.sigmay_tensor_name,feed_dict=feed_data)
            sigmax_tensor  = [np.transpose(np.concatenate((sigmax, self.zero_3d), axis=2), [0,2,1]) for sigmax in sigmax_tensor]
            sigmay_tensor  = [np.transpose(np.concatenate((sigmay, self.zero_3d), axis=2), [0,2,1]) for sigmay in sigmay_tensor]
            return sigmax_tensor, sigmay_tensor
        if tensor_name == 'SUBPROB':
            subprob_tensor = self.sess.run(self.subprob_tensor_name,feed_dict=feed_data)
            subprob_tensor = [np.concatenate((subprob, self.zero_3d), axis=2) for subprob in subprob_tensor]
            subprob_tensor = [np.transpose(subprob,[0,2,1]).reshape((-1,self.component_num)) for subprob in subprob_tensor]
            return subprob_tensor
        if tensor_name == 'PROB':
            prob_tensor = self.sess.run(self.prob_tensor_name,feed_dict=feed_data)
            prob_tensor = [np.concatenate((prob, self.zero_2d), axis=1).reshape((-1,1)) for prob in prob_tensor]
            return prob_tensor
        if tensor_name == 'PI':
            pi_tensor = self.sess.run(self.pi_tensor_name,feed_dict=feed_data)
            pi_tensor = [np.concatenate((pi, self.zero_3d), axis=2) for pi in pi_tensor]
            pi_tensor = [np.transpose(pi,[0,2,1]).reshape((-1,self.component_num)) for pi in pi_tensor]
            return pi_tensor
        if tensor_name == 'RHO':
            rho_tensor = self.sess.run(self.rho_tensor_name,feed_dict=feed_data)
            rho_tensor = [np.concatenate((rho, self.zero_3d), axis=2) for rho in rho_tensor]
            rho_tensor = [np.transpose(rho,[0,2,1]).reshape((-1,self.component_num)) for rho in rho_tensor]
            return rho_tensor

