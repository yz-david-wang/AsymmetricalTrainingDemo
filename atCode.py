
import numpy as np
import tensorflow as tf
from scipy.special import softmax

def relu_act(x, x_0=0.0):
    return np.maximum(x_0, x)

def relu_de(x, x_0=0.0):
    return np.where(x > x_0, 1, 0)

def abs_error(pred, y):
    return pred - y

class NNTrain:
    def __init__(self, layer_structure):
        self.layer_structure = layer_structure
        self.W_M = []
        self.bias = []
        self.a = None
        self.z = None
        self.no_layer_param = len(layer_structure) - 1

    def initial_param(self, base_seed=42):
        for i in range(self.no_layer_param):
            np.random.seed(base_seed + i)
            self.W_M.append(0.1*np.random.randn(self.layer_structure[i+1], self.layer_structure[i]))
        for i in range(self.no_layer_param):
            np.random.seed(base_seed + i)
            self.bias.append(0.5*np.random.randn(self.layer_structure[i+1]))
        np.random.seed(None)

    @staticmethod
    def error(pred, y, method=abs_error):
        error = method(pred, y)
        return error

    @staticmethod
    def get_accuracy(pred, y):
        classify = pred
        score = 0
        for j in range(pred.shape[1]):
            if np.argmax(classify[:, j]) == np.argmax(y[:, j]):
                score += 1
        return score/pred.shape[1]

    def __repr__(self):
        return f"NNTrain ({self.layer_structure})"


class AsymmetricTrain(NNTrain):
    def __init__(self, layer_structure, w_dev=60, p_dev=60, gen_seed=42):
        super().__init__(layer_structure)
        self.w_dev = w_dev
        self.w_mod = None
        self.b_mod = None
        self.w_phy = None
        self.b_phy = None
        self.p_dev = p_dev
        self.dev_seed = gen_seed
        self.b_dev = None
        self.dw_er_mask = None
        self.db_er_mask = None
        self.phy_dig_sep_w = None
        self.phy_dig_sep_b = None
        self.x_test = None
        self.y_test = None

    def param_dev_init(self):
        self.w_mod = self.W_M.copy()
        self.b_mod = self.bias.copy()
        for i in range(len(self.W_M)):
            np.random.seed(self.dev_seed * (2*i+1))
            self.w_mod[i] =\
                self.w_mod[i] + self.w_mod[i] * (np.random.randn(self.w_mod[i].shape[0],
                                                                 self.w_mod[i].shape[1])) * (self.w_dev/100)
            self.b_mod[i] =\
                self.b_mod[i] + self.b_mod[i] * (self.w_dev/100) * np.random.randn(self.b_mod[i].shape[0])

    def p_dev_init(self):
        self.dw_er_mask = self.W_M.copy()
        self.db_er_mask = self.bias.copy()
        for i in range(self.no_layer_param):
            np.random.seed(self.dev_seed * (3*i+3))
            self.dw_er_mask[i] = (self.p_dev/100) * (np.random.randn(self.dw_er_mask[i].shape[0],
                                                                     self.dw_er_mask[i].shape[1]))
            self.db_er_mask[i] = (self.p_dev/100) * (np.random.randn(self.db_er_mask[i].shape[0]))
    
    @staticmethod
    def forwardDig(x, w_m, bias, activation=relu_act):
        # Forming the forward pass in silico
        a = [x]
        z = []
        for layer in range(len(w_m) - 1):
            net = (w_m[layer] @ a[layer])
            for j in range(x.shape[1]):
                net[:, j] += bias[layer]
            out = activation(net)
            z.append(net)
            a.append(out)
        net = w_m[-1] @ a[-1]
        for j in range(x.shape[1]):
            net[:, j] += bias[-1]
        z.append(net)
        out = softmax(net, axis=0)
        a.append(out)
        return a, z, a[-1]
    
    @staticmethod
    def forwardPara(x, w_m, bias, lay_struct, activation=relu_act):
        # The forward pass of the digital parallel model
        a = [x]
        z = []
        for layer in range(len(w_m) - 1):
            net = (w_m[layer] @ a[layer])
            for j in range(x.shape[1]):
                net[:, j] += bias[layer]
            net = np.clip(net, -5*lay_struct[layer], 5*lay_struct[layer])
            out = activation(net)
            z.append(net)
            a.append(out)
        net = w_m[-1] @ a[-1]
        for j in range(x.shape[1]):
            net[:, j] += bias[-1]
        z.append(net)
        out = softmax(net, axis=0)
        a.append(out)
        return a, z, a[-1]
    
    @staticmethod
    def forwardPhy(x, w_m, bias, lay_struct, activation=relu_act, fd = 0.0):
        # The physical propagation
        a = [x]
        z = []
        errRate = fd
        np.random.seed(None)
        for layer in range(len(w_m) - 1):
            net = (w_m[layer] @ a[layer])
            for j in range(x.shape[1]):
                net[:, j] += bias[layer]
            net = (1 + np.random.randn(1) * errRate) * net
            net = np.clip(net, -5*lay_struct[layer], 5*lay_struct[layer])
            out = activation(net)
            z.append(net)
            a.append(out)
        net = w_m[-1] @ a[-1]
        for j in range(x.shape[1]):
            net[:, j] += bias[-1]
        net = (1 + np.random.randn(1) * errRate) * net
        z.append(net)
        out = softmax(net, axis=0)
        a.append(out)
        return a, z, a[-1]

    @staticmethod
    def bp_back_gen(a, z, w_m, out, error, no_layer_param, act_de=relu_de):
        # The general backwards pass of BP
        d_w = []
        d_b = []
        d_z = [error]
        d_b.append(np.sum(d_z[0], axis=1))
        d_w.append(d_z[0] @ a[-2].T)
        for i in range(1, no_layer_param, 1):
            d_z.append(w_m[-i].T @ d_z[-1] * act_de(z[-(i + 1)]))
            d_b.append(np.sum(d_z[-1], axis=1))
            d_w.append(d_z[-1] @ a[-(i + 2)].T)
        d_w = d_w[::-1]
        d_b = d_b[::-1]
        return d_w, d_b, d_z

    @staticmethod
    def update_param_gen(no_layer_param, w_m, bias, d_w, d_b, l_r):
        for i in range(no_layer_param):
            w_m[i] = w_m[i] - (l_r * d_w[i])
            bias[i] = bias[i] - (l_r * d_b[i])

    @staticmethod
    def projection_err(d_w, d_b, d_w_err, d_b_err):
        for i in range(len(d_w)):
            d_w[i] = d_w[i] * d_w_err[i] + d_w[i]
            d_b[i] = d_b[i] * d_b_err[i] + d_b[i]
        return d_w, d_b

    def phy_dig_sep_proj(self):
        self.w_phy = self.W_M.copy()
        self.b_phy = self.bias.copy()
        for i in range(self.no_layer_param):
            self.w_phy[i] = self.w_phy[i] * self.phy_dig_sep_w[i] + self.w_phy[i]
            self.b_phy[i] = self.b_phy[i] * self.phy_dig_sep_b[i] + self.b_phy[i]

    def phy_dig_sep_mask(self):
        np.random.seed(self.dev_seed)
        self.phy_dig_sep_w = self.W_M.copy()
        self.phy_dig_sep_b = self.bias.copy()
        for i in range(self.no_layer_param):
            self.phy_dig_sep_w[i] = (self.w_dev/100) * np.random.randn(self.phy_dig_sep_w[i].shape[0], self.phy_dig_sep_w[i].shape[1])
            self.phy_dig_sep_b[i] = (self.w_dev/100) * (np.random.randn(self.phy_dig_sep_b[i].shape[0]))

    def phy_dig_sep_proj2(self, w_phy, b_phy):
        for i in range(self.no_layer_param):
            self.w_phy[i] = w_phy[i] * self.phy_dig_sep_w[i] + w_phy[i]
            self.b_phy[i] = b_phy[i] * self.phy_dig_sep_b[i] + b_phy[i]

    @staticmethod
    def sep_weighting(weighting, update):
        for i in range(len(update)):
            update[i] = update[i] * weighting
        return update

    def take_test_data(self, x_t, y_t):
        self.x_test = x_t
        self.y_test = y_t
        self.x_test = np.transpose(self.x_test)
        self.y_test = np.transpose(self.y_test)

    def atMethod(self, x, y, epochs=100, batch_size=200, l_r=1e-3, m_r1=0.5, m_r2=0.5):
        x = np.transpose(x)
        y = np.transpose(y)
        num_of_batch = x.shape[1] // batch_size
        w_phy = self.W_M.copy()
        b_phy = self.bias.copy()
        acc_list = []
        acc_list_test = []
        self.phy_dig_sep_mask()
        self.phy_dig_sep_proj()
        self.phy_dig_sep_proj2(self.W_M, self.bias)
        for i in range(epochs):
            shuffle = np.random.permutation(x.shape[1])
            x = x[:, shuffle]
            y = y[:, shuffle]
            loss = 0.0
            for j in range(num_of_batch):
                sample = x[:, j * batch_size:(j + 1) * batch_size]
                target = y[:, j * batch_size:(j + 1) * batch_size]
                a, z, out_id = self.forwardPara(sample, self.W_M, self.bias, self.layer_structure)
                _, _, out = self.forwardPhy(sample, self.w_phy, self.b_phy, self.layer_structure)
                err1 = self.error(out_id, target)
                d_w1, d_b1, _ = self.bp_back_gen(a, z, self.W_M, out_id, err1, self.no_layer_param)
                err2 = self.error(out, target)
                d_w2, d_b2, _ = self.bp_back_gen(a, z, self.W_M, out, err2, self.no_layer_param)
                d_w1m = self.sep_weighting(m_r1, d_w1)
                d_b1m = self.sep_weighting(m_r1, d_b1)
                d_w2 = self.sep_weighting(m_r2, d_w2)
                d_b2 = self.sep_weighting(m_r2, d_b2)
                d_w1m, d_b1m = self.projection_err(d_w1m, d_b1m, self.dw_er_mask, self.db_er_mask)
                d_w2, d_b2 = self.projection_err(d_w2, d_b2, self.dw_er_mask, self.db_er_mask)
                self.update_param_gen(self.no_layer_param, self.W_M, self.bias, d_w1, d_b1, l_r)
                self.update_param_gen(self.no_layer_param, w_phy, b_phy, d_w1m, d_b1m, l_r)
                self.update_param_gen(self.no_layer_param, w_phy, b_phy, d_w2, d_b2, l_r)
                self.phy_dig_sep_proj2(w_phy, b_phy)
                batch_loss = np.array(tf.compat.v1.losses.log_loss(target, out))
                loss += batch_loss
            _, _, pred = self.forwardPhy(x, self.w_phy, self.b_phy, self.layer_structure)
            train_accuracy = self.get_accuracy(pred, y)
            acc_list.append(train_accuracy)
            _, _, pred2 = self.forwardPhy(self.x_test, self.w_phy, self.b_phy, self.layer_structure)
            test_accuracy = self.get_accuracy(pred2, self.y_test)
            acc_list_test.append(test_accuracy)
            print("Train accuracy is ", train_accuracy)
            print("Test accuracy is ", test_accuracy)
            print("Loss at epoch", i + 1, "is", loss / x.shape[1])
            if np.isnan(loss / x.shape[1]) == True:
                break
        return acc_list, acc_list_test

    def inscilicoBP(self, x, y, epochs=100, batch_size=200, l_r=1e-3):
        x = np.transpose(x)
        y = np.transpose(y)
        num_of_batch = x.shape[1] // batch_size
        acc_list = []
        acc_list_test = []
        self.phy_dig_sep_mask()
        self.phy_dig_sep_proj()
        self.phy_dig_sep_proj2(self.W_M, self.bias)
        for i in range(epochs):
            shuffle = np.random.permutation(x.shape[1])
            x = x[:, shuffle]
            y = y[:, shuffle]
            loss = 0.0
            for j in range(num_of_batch):
                sample = x[:, j * batch_size:(j + 1) * batch_size]
                target = y[:, j * batch_size:(j + 1) * batch_size]
                a, z, out_id = self.forwardDig(sample, self.W_M, self.bias)
                err = self.error(out_id, target)
                d_w, d_b, _ = self.bp_back_gen(a, z, self.W_M, out_id, err, self.no_layer_param)
                d_w2, d_b2 = self.projection_err(d_w, d_b, self.dw_er_mask, self.db_er_mask)
                self.update_param_gen(self.no_layer_param, self.W_M, self.bias, d_w2, d_b2, l_r)
                self.phy_dig_sep_proj()
                self.phy_dig_sep_proj2(self.W_M, self.bias)
                _, _, out_phy = self.forwardPhy(sample, self.w_phy, self.b_phy, self.layer_structure)
                batch_loss = np.array(tf.compat.v1.losses.log_loss(target, out_phy))
                loss += batch_loss
            _, _, pred = self.forwardPhy(x, self.w_phy, self.b_phy, self.layer_structure)
            train_accuracy = self.get_accuracy(pred, y)
            acc_list.append(train_accuracy)
            _, _, pred2 = self.forwardPhy(self.x_test, self.w_phy, self.b_phy, self.layer_structure)
            test_accuracy = self.get_accuracy(pred2, self.y_test)
            acc_list_test.append(test_accuracy)
            print("Train accuracy is ", train_accuracy)
            print("Test accuracy is ", test_accuracy)
            print("Loss at epoch", i + 1, "is", loss / x.shape[1])
            if np.isnan(loss / x.shape[1]) == True:
                break
        return acc_list, acc_list_test

    def __repr__(self):
        return f"Asymmetrical Train (Layer structure = {self.layer_structure}, param dev = {self.w_mod}," \
               f" proj dev = {self.p_dev})"

