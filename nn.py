import math
import random
import numpy as np


learning_rate = 0.01
momentum_rate = 0.2
dropout_rate = 0.0
weight_decay_rate = 0.0


# xor
xx = [(1,0), (0,1), (1,1), (0,0)]
yy  = [1,1,0,0]
# xx = [(1,0), (0,1), (1,1), (0,0)]
# yy  = [1,1,0,0]

# and
# xx = [(1,0), (0,1), (1,1), (0,0)]
# yy  = [1,1,0,0]

# or
# xx = [(1,-1), (-1,1), (1,1), (-1,-1)]
# yy  = [1,1,1,0]


# xx = [(1,1), (1,1)]
# yy  = [1, 1]


train_set = zip(xx, yy)


def sig(x):
    # return 1.0 / (1 + math.exp(-x))
    if x > 0:
        return x
    # import pudb; pudb.set_trace()  # NOQA  # pylint: disable=multiple-statements
    return 0


def prime_sig(x):
    # return sig(x) * (1.0 - sig(x))
    if x > 0:
        return 1
    return 0


def cost(y, a):
    return (y - a) * (y - a)


def gradient_cost(y, a):
    return 2 * (a - y)


class Node:
    def __init__(self, batch_size=10):
        self.v = [-1.0 for _ in range(batch_size)]  # value (= output)
        self.p = [0.0 for _ in range(batch_size)]  # prime value
        self.drop = [False for _ in range(batch_size)]  # dropout


class Layer:
    def __init__(self, node_cnt):
        self.node_cnt = -1
        self.nodes = []
        self.node_cnt = node_cnt
        for i in range(node_cnt):
            self.nodes.append(Node())


class NN:
    def __init__(self, input_layer_node_cnt):
        self.layers = []
        self.layers.append(Layer(input_layer_node_cnt))
        self.weights = [[]]
        self.bias = [[]]
        self.pv_cost = 10000000000
        self.online = False
        self.dropout = False

    def add_layer(self, node_cnt, default_weight=0.5):
        assert len(self.layers) > 0
        pl = self.layers[len(self.layers) - 1]
        self.layers.append(Layer(node_cnt))
        nl = self.layers[len(self.layers) - 1]
        ws = [[] for _ in range(len(nl.nodes))]
        for i in range(len(ws)):
            # ws[i] = [default_weight for _ in range(len(pl.nodes))]
            ws[i] = [
                (random.random() * 0.5) * 2.0 for _ in range(len(pl.nodes))]
        self.weights.append(ws)
        # self.bias.append(
            # [(random.random() * 0.5) * 2.0 for _ in range(len(nl.nodes))])
        self.bias.append([0 for _ in range(len(nl.nodes))])

    # change p-value of final layer's nodes to p-value * cost gradient
    def _multiply_cost_gradient_to_last_layer(self, batch_index, x, y):
        fl = self.layers[-1]
        for n in fl.nodes:
            # gradient_cost can be interpreted as error_rate
            error_rate = gradient_cost(y, n.v[batch_index])
            n.p[batch_index] *= error_rate

    def _backword(self, batch_set):  # backpropagation
        # update p-values
        for i in range(len(self.layers) - 1):
            li = -(i + 1)
            ll = self.layers[li - 1]
            rl = self.layers[li]
            ws = self.weights[li]
            for bi in range(len(batch_set)):
                for fr in range(len(ll.nodes)):
                    if ll.nodes[fr].drop[bi] is True: continue
                    error_rate_sum = 0  # sum over all nodes
                    for to in range(len(rl.nodes)):
                        if rl.nodes[to].drop[bi] is True: continue
                        rp = rl.nodes[to].p[bi]
                        error_rate_sum += ws[to][fr] * rp
                    ll.nodes[fr].p[bi] *= error_rate_sum

        # update w and b
        for i in range(len(self.layers) - 1):
            li = -(i + 1)
            ll = self.layers[li - 1]
            rl = self.layers[li]
            ws = self.weights[li]
            #print('ws before:{}'.format(ws))
            for fr in range(len(ll.nodes)):
                if ll.nodes[fr].drop[bi] is True: continue
                for to in range(len(rl.nodes)):
                    if rl.nodes[to].drop[bi] is True: continue
                    delta_nabla_w = 0
                    for bi in range(len(batch_set)):
                        lv = ll.nodes[fr].v[bi]
                        rp = rl.nodes[to].p[bi]
                        delta_nabla_w += learning_rate * lv * rp\
                            + momentum_rate * delta_nabla_w
                    total_set_len = len(batch_set)  # FIXME should be fixed
                    ws[to][fr] = (1.0 - learning_rate * weight_decay_rate / total_set_len) * ws[to][fr] - delta_nabla_w / len(batch_set)
            for to in range(len(rl.nodes)):
                if rl.nodes[to].drop[bi] is True: continue
                delta_nabla_b = 0
                for bi in range(len(batch_set)):
                    rp = rl.nodes[to].p[bi]
                    delta_nabla_b += learning_rate * rp
                self.bias[li][to] -= delta_nabla_b / len(batch_set)
            #print('ws after:{}'.format(ws))

    def __forward(self, batch_index, flag_test=False):
        bi = batch_index
        for i in range(1, len(self.layers)):
            pl = self.layers[i - 1]
            nl = self.layers[i]
            for to in range(len(nl.nodes)):
                if nl.nodes[to].drop[bi] is True: continue
                sum = 0
                for fr in range(len(pl.nodes)):
                    if pl.nodes[fr].drop[bi] is True: continue
                    n = pl.nodes[fr].v[bi] * self.weights[i][to][fr]
                    if self.dropout and flag_test is True: n *= 1.0 - dropout_rate
                    sum += n
                sum += self.bias[i][to]
                nl.nodes[to].v[bi] = sig(sum)
                nl.nodes[to].p[bi] = prime_sig(sum)

    def _quadratic_weight_sum(self):
        sum = 0
        l1 = len(self.weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    w = self.weights[i][j][k]
                    sum += w * w
        # n1 = np.linalg.norm(self.weights)**2
        # assert n1 == sum
        return sum

    def _forward(self, batch_set):
        if self.dropout is True:
            # except input and output layer
            for i in range(1, len(self.layers) - 1):
                l = self.layers[i]
                r = None
                while True:
                    r = list(np.random.binomial(1, 1.0 - dropout_rate, size=len(l.nodes)))
                    if r.count(1) > 0:
                        break

                print r
                for ix in range(len(l.nodes)):
                    for bi, (x, y) in enumerate(batch_set):
                        if r[ix] == 0:
                            l.nodes[ix].drop[bi] = True
                        else:
                            l.nodes[ix].drop[bi] = False

        c_sum = 0
        for bi, (x, y) in enumerate(batch_set):
            input_layer = self.layers[0]
            assert len(x) == len(input_layer.nodes)
            for i in range(len(input_layer.nodes)):
                input_layer.nodes[i].v[bi] = x[i]
            self.__forward(bi)
            fl = self.layers[-1]
            for n in fl.nodes:
                c_sum += cost(y, n.v[bi])
            if weight_decay_rate > 0:
                c_sum += weight_decay_rate * self._quadratic_weight_sum()
            self._multiply_cost_gradient_to_last_layer(bi, x, y)
        c_sum /= len(batch_set)
        print 'cost: {}'.format(c_sum)
        # assert c_sum <= self.pv_cost
        self.pv_cost = c_sum

    def train(self, batch_set, iter_cnt, **kwargs):
        if 'online' in kwargs:
            self.online = kwargs['online']
        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']

        print 'w before: {}'.format(self.weights)
        print 'b before: {}'.format(self.bias)
        for i in range(iter_cnt):
            if self.online:
                for sample in batch_set:
                    self._forward([sample])
                    self._backword([sample])
            else:
                self._forward(batch_set)
                self._backword(batch_set)

    def _test(self, x):
        self.__forward(0, True)
        fl = self.layers[-1]
        for i in range(len(fl.nodes)):
            a = fl.nodes[i].v[0]
            print ('-=-=-=-= test output: {}'.format(a))

    def test(self, x):
        print 'test..x:{}'.format(x)
        # print 'w:{}'.format(self.weights)
        # print 'b:{}'.format(self.bias)
        input_layer = self.layers[0]
        assert len(x) == len(input_layer.nodes)
        for i in range(len(input_layer.nodes)):
            input_layer.nodes[i].v[0] = x[i]
        self._test(x)

nn = NN(len(train_set[0][0]))  # create nn with input layer of node cnt 2
nn.add_layer(2)  # add output layer
nn.add_layer(1)  # add output layer

# print y
# nn.weights = [[], [[1.0, -1.0], [-1.0, 1.0]], [[1.0, 1.0]]]
# nn.bias = [[], [-0.5, -0.5], [-0.5]]
nn.train(train_set, 20000, online=False, dropout=False)  # train with train set(Y) and iteration cnt
for xxx in xx:
    nn.test(xxx)
w = nn.weights
print nn.weights
print nn.bias
