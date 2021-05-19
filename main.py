import os
from time import sleep
import time
from numpy.core.fromnumeric import prod
from tqdm import tqdm
import numpy as np
from typing import List
import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import itertools


def binary_space_addition(*bits):
    """Takes list of bits and sums them modulo two (preserves erased bits)"""
    return sum([n for n in bits]) % 2


def all_erased(*bits):
    """Checks if all bits are erased in a bit"""
    for bit in bits:
        if not bit.is_erased():
            return False
    else:
        return True


def random_non_erased_bit(*bits):
    # TODO: Fix this what is this even
    """From a list of bits, randomly selects one that is not erased"""
    non_erased_bits = [b for b in bits if not b.is_erased()]
    return random.choice(non_erased_bits)


def conditional_print(message, do):
    """util"""
    if do:
        print(message)


def binary_mat_mul(a, b):
    """Binary matrix multiplication - check dimension constraints?"""
    dec = np.matmul(a, b)
    out = [binary_space_addition(j) for j in dec]
    return out


class Bit:

    """
    Bits can take one of three values - 0, 1, nan (erased)
    They can be added & multiplied with other bits using the normal python operators +, *
    """

    def __init__(self, value: float):
        if value not in {0, 1} and not np.isnan(value.real):
            raise ValueError(f"Bit must be of value 0, 1, or nan, not {value}")
        self.real = value

    def erase(self):
        """Erased the bit"""
        self.real = np.nan

    def is_erased(self):
        """Check if bit is erased"""
        return np.isnan(self.real)

    def __add__(self, other):
        """This and all below are just util functions"""
        bit_value = binary_space_addition(self.real + other.real)
        return Bit(bit_value)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        bit_value = self.real * other.real
        return Bit(bit_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        return Bit(self.real%other.real)

    def __repr__(self):
        if self.is_erased():
            return f"b?"
        else:
            return f"b{self.real}"

    def __str__(self):
        if self.real == 1:
            return '1'
        elif self.real == 0:
            return '0'
        elif np.isnan(self.real):
            return '?'
        else:
            raise ValueError('How did you even get here? Bit valued ' + str(self.real))


class AWGNChannel:

    """
    This is a (symmetric) AWGN only! Need to write entirely new code for other channel formations
    """

    def __init__(self, sigma: float):
        self.sigma = sigma
        self.alphabet = {0:1, 1:-1}

    def transmit_message(self, bits: List[Bit]):

        channel_output = [self.alphabet[bit.real] + np.random.randn() * self.sigma for bit in bits]
        channel_beliefs = [2*co*(self.sigma**-2) for co in channel_output]

        return channel_beliefs


class Channel:

    """This is a (symmetric) BEC only! Need to write entirely new code for other channel formations"""

    def __init__(self, p: float):
        self.p = p
        self.channel_beliefs = {0: np.inf, 1: -np.inf, np.nan:0}

    def transmit_message(self, bits: List[Bit]):
        out_bits = [Bit(b.real) for b in bits]
        for bit in out_bits:
            if random.random() < self.p:
                bit.erase()

        channel_beliefs = [self.channel_beliefs[ob.real] for ob in out_bits]
        return channel_beliefs


class VariableCheckEdge:

    """
    The edge between a variable and check node in the LDPC bipartite graph
    This stores the most recently passed vc and cv messages, which can be accessed by the nodes & by the visualisation code
    """

    def __init__(self, v_node, c_node, v_idx, c_idx):
        self.cv_message = 0
        self.vc_message = 0
        self.v_node = v_node
        self.c_node = c_node
        v_node.add_edge(c_idx, self)
        c_node.add_edge(v_idx, self)

    def __repr__(self):
        return f"|CV:{self.cv_message.__repr__()}|VC:{self.vc_message.__repr__()}|"


class CheckNode:

    def __init__(self):
        self.edges = {}

    def add_edge(self, v_idx, edge: VariableCheckEdge):
        self.edges[v_idx] = edge

    def update_messages(self):
        """
        CV message is a function of all incoming messages *other* than the one it is sending to
        """
        for v_idx, edge in self.edges.items():
            # node_inputs = [edge.vc_message for v_dash, edge in self.edges.items() if v_dash != v_idx]
            # edge.cv_message = binary_space_addition(*node_inputs)
            node_inputs = [edge.vc_message for v_dash, edge in self.edges.items() if v_dash != v_idx]
            inner_tanh = prod([np.tanh(0.5*n) for n in node_inputs])
            edge.cv_message = 2*np.arctanh(inner_tanh)


    def __repr__(self):
        s = "CNode: "
        for edge in self.edges.values():
            s += " " + edge.cv_message.__repr__()
        return s


class VariableNode:

    def __init__(self):
        self.edges = {}

    def update_channel_belief(self, channel_belief):
        self.channel_belief = channel_belief

    def add_edge(self, c_idx, edge: VariableCheckEdge):
        self.edges[c_idx] = edge

    def update_messages(self):
        for c_idx, edge in self.edges.items():
            node_inputs = [edge.cv_message for c_dash, edge in self.edges.items() if c_dash != c_idx]
            edge.vc_message = sum(node_inputs) + self.channel_belief

    def loglikelihood(self):
        node_inputs = sum([edge.cv_message for c_dash, edge in self.edges.items()])
        return node_inputs + self.channel_belief

    def __repr__(self):
        return "VNode: " + str(self.value.__repr__())


class Encoder:

    def __init__(self, g_sys):
        self.g_sys = g_sys
        self.k = g_sys.shape[0]
        self.n = g_sys.shape[1]
        self.r = self.k / self.n

    def __call__(self, message: List[int]):
        code_bits = binary_mat_mul(message, self.g_sys)
        return code_bits


class Decoder:

    def __init__(self, parity_matrix, decoderbook):

        """
        Get dimensions, initialise the right number of each type of node,
            call form_tree(), then run asserts
        """

        self.decoderbook = decoderbook
        self.parity_matrix = parity_matrix
        self.n = parity_matrix.shape[1]
        self.k = self.n - parity_matrix.shape[0]
        self.r = self.k / self.n
        self.message = [Bit(np.nan) for _ in range(self.n)]
        self.check_nodes = [CheckNode() for _ in range(self.parity_matrix.shape[0])]
        self.variable_nodes = [VariableNode() for _ in range(self.n)]
        self.edges = np.zeros(self.parity_matrix.shape, dtype = object)
        self.t_max = 100
        self.form_tree()

        for c in range(self.parity_matrix.shape[0]):
            for v in range(self.parity_matrix.shape[1]):
                if self.edges[c,v]:
                    assert self.variable_nodes[v].edges[c] == self.check_nodes[c].edges[v]
                    assert self.edges[c,v] == self.check_nodes[c].edges[v]
                    assert self.edges[c, v] == self.variable_nodes[v].edges[c]

    def form_tree(self):
        """
        Store the correct edges in the self.edges matrix
        """
        for c in range(self.parity_matrix.shape[0]):
            for v in range(self.parity_matrix.shape[1]):
                if self.parity_matrix[c, v]:
                    check_node = self.check_nodes[c]
                    variable_node = self.variable_nodes[v]
                    self.edges[c,v] = VariableCheckEdge(v_node=variable_node, c_node=check_node, v_idx=v, c_idx=c)

    def update_check_nodes(self):
        for check_node in self.check_nodes:
            check_node.update_messages()

    @staticmethod
    def codeword_distance(a, b):
        return sum(np.array(list(a)) != np.array(list(b)))

    def find_closest_message(self, decoderbook):
        decoderbook = "".join([str(b.real) for b in decoderbook])
        closest_codeword = min(self.decoderbook.keys(), key=lambda x: self.codeword_distance(x, decoderbook))
        return self.decoderbook[closest_codeword]

    def make_blank_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(["c"+str(c) for c in range(len(self.check_nodes))], bipartite=0, color='r')
        graph.add_nodes_from(["v"+str(v) for v in range(len(self.variable_nodes))], bipartite=1, color='orange')
        return graph

    def draw_colored_edge(self, _c, _v, _graph, mode):
        if mode == "cv":
            _message = self.edges[_c, _v].cv_message.__repr__()
        elif mode == "vc":
            _message = self.edges[_c, _v].vc_message.__repr__()
        colourbook = {"b1": 'green', "b0": "red", "b?": "gray"}
        _graph.add_edge("c"+str(_c), "v"+str(_v), color=colourbook[_message])

    def save_graph(self, _graphs, _titles, _filename):
        fig, axs = plt.subplots(1, len(_graphs))
        for i, _g in enumerate(_graphs):
            l, r = nx.bipartite.sets(_g)
            pos = {}
            pos.update({node: (1, index) for index, node in enumerate(l)})
            pos.update({node: (2, index) for index, node in enumerate(r)})
            nx.draw(_g, pos=pos, ax=axs[i], edge_color=[_g.edges[edge]['color'] for edge in _g.edges()], node_color=[_g.nodes[node]['color'] for node in _g.nodes()])
            axs[i].set_title(_titles[i])
        fig.savefig(_filename)
        plt.close(fig)

    def make_graphic(self, t, dir):
        return None # remove when ready
        filename = os.path.join(dir, f"round-{t}.png")

        cv_graph = self.make_blank_graph()
        vc_graph = self.make_blank_graph()

        for c in range(self.parity_matrix.shape[0]):
            for v in range(self.parity_matrix.shape[1]):
                if self.parity_matrix[c, v]:
                    self.draw_colored_edge(c, v, cv_graph, "cv")
                    self.draw_colored_edge(c, v, vc_graph, "vc")

        self.save_graph([cv_graph, vc_graph], ["CV", "VC"], filename)

    def decode_message(self, channel_beliefs: List[Bit], real_message: List[Bit]):
        assert len(channel_beliefs) == self.n

        # root_dir = "".join([str(b.real)[0] for b in real_message]) + "--" + "".join([str(b.real)[0] for b in input_message]) + "--" + str(time.time())
        # os.mkdir(root_dir)

        for c in range(len(self.check_nodes)):
            for v in range(len(self.variable_nodes)):
                # Reset actual edges
                if self.edges[c,v]:
                    self.edges[c,v].cv_message = 0
                
        for j, cb in enumerate(channel_beliefs):
            self.variable_nodes[j].channel_belief = cb


        t = 0
        # self.make_graphic(t, root_dir)
        while t < self.t_max:
            for v, variable_node in enumerate(self.variable_nodes):
                variable_node.update_messages() #(input_message[v])
            self.update_check_nodes()
            t += 1
            # self.make_graphic(t, root_dir)

        likelihoods = [v.loglikelihood() for v in self.variable_nodes]
        print("Final LLHs: ", likelihoods)

        return [int(l < 0) for l in likelihoods]


class Pipeline:

    def __init__(self, g_sys, channel):
        parity_matrix, n, k = self.make_parity_matrix(g_sys)
        self.encoder = Encoder(g_sys)
        decoderbook = self.construct_decoderbook(g_sys)
        self.decoder = Decoder(parity_matrix, decoderbook)
        self.channel = channel
        assert self.encoder.r == self.decoder.r

    @staticmethod
    def make_parity_matrix(g_sys):
        assert len(g_sys.shape) == 2 and all(np.unique(g_sys) == np.array([0, 1]))
        n = g_sys.shape[1]
        k = g_sys.shape[0]
        assert k < n
        if not np.array_equal(g_sys[:,:k], np.eye(k)):
            raise NotImplementedError("Generator matrix must be systematic!")
        P = g_sys[:,k:]
        parity_matrix = np.concatenate([P, np.eye(n - k)]).T
        return parity_matrix, n, k

    def construct_decoderbook(self, g_sys):
        all_messages = [list(a) for a in itertools.product([0, 1], repeat=g_sys.shape[0])]
        stringify = lambda x: "".join(str(a) for a in x)
        decoderbook = {stringify(self.encoder(msg)): stringify(msg) for msg in all_messages}
        return decoderbook

    def __call__(self, _message: List[int], show=True):
        
        # The k message bits
        conditional_print(f"Input message bits: \t {_message}", do=show)
        _message = [Bit(int(_m)) for _m in _message]

        # The n coded bits
        channel_input_code_bits = self.encoder(_message)
        conditional_print(f"Channel input code bits: \t {channel_input_code_bits}", do=show)

        # The n channel beliefs [L(y_j) in the databook]
        channel_beliefs = self.channel.transmit_message(channel_input_code_bits)
        conditional_print(f"Channel output code bits: \t {channel_beliefs}", do=show)

        # Varaible node loglikelihoods after LDPC message passing algorithm
        recovered_code = self.decoder.decode_message(channel_beliefs, channel_input_code_bits)
        conditional_print(f"Recovered code bits: \t {recovered_code}", do=show)
      
        # Use codebook to get back the closest message
        decoded_message = self.decoder.find_closest_message(recovered_code)
        conditional_print(f"Decoded final output: \t {decoded_message}", do=show)

        return decoded_message


if __name__ == '__main__':

    codes = ["".join(str(j) for j in i) for i in list(itertools.product([0, 1], repeat=5))][:4]

    generator_matrix = np.array(
        [[1, 0, 0, 0, 1, 1, 0],
         [0, 1, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]]
    )

    generator_matrix = np.array(
        [[1,0,0,0,0,0,0,1,0,1],
        [0,1,0,0,0,1,0,0,0,1],
        [0,0,1,0,0,1,1,0,1,0],
        [0,0,0,1,0,1,1,0,1,1],
        [0,0,0,0,1,1,1,0,1,0]]
    )
    sigmas = np.linspace(0.5,3,500)
    final_error_rates = []

    for sigma in tqdm(sigmas):
        channel = AWGNChannel(sigma)
        pipeline = Pipeline(g_sys=generator_matrix, channel = channel)
        
        error_count = 0
        total_count = 0
        
        for code in codes:
            
            output = pipeline(code)
            error_count += sum(np.array(list(code)) != np.array(list(output)))
            total_count += len(code)

        final_error_rates.append(error_count/total_count)    
        

    plt.plot(sigmas, final_error_rates)
    plt.savefig("asdf.png")

    while True:
        channel = AWGNChannel(sigma)
        pipeline = Pipeline(g_sys=generator_matrix, channel=channel)
        message = input(f"Please input message, of size {pipeline.encoder.k} in form 1101001...: ")
        message = [int(a) for a in message]
        pipeline(message, show=True)
        print("\n")