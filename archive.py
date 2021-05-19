class VariableNode:

    def __init__(self, channel_belief):
        self.edges = {}
        self.channel_belief = channel_belief

    def add_edge(self, c_idx, edge: VariableCheckEdge):
        self.edges[c_idx] = edge

    def change_value(self, new_value):
        try:
            if self.value.is_erased():
                self.value = Bit(new_value)
        except:
            import pdb; pdb.set_trace()

    def update_messages(self, channel_input: Bit):
        # TODO: generalise to allow LLR form
        if not channel_input.is_erased():
            for edge in self.edges.values():
                edge.vc_message = Bit(value=channel_input.real)
            self.value = Bit(channel_input.real)

        else:
            for c_idx, edge in self.edges.items():
                check_node_inputs = [edge.cv_message for c_dash, edge in self.edges.items() if c_dash != c_idx]
                if all_erased(*check_node_inputs):
                    edge.vc_message = Bit(value=np.nan)
                else:
                    #bit_value = random_non_erased_bit(*check_node_inputs).real
                    bit_value = [c for c in check_node_inputs if not c.is_erased()][0].real
                    edge.vc_message = Bit(value=bit_value)

            all_incoming_checks = [edge.cv_message for c_dash, edge in self.edges.items() if not edge.cv_message.is_erased()]
            if len(all_incoming_checks):
                self.change_value(all_incoming_checks[0].real)

    def __repr__(self):
        return "VNode: " + str(self.value.__repr__())


class Channel:

    """This is a (symmetric) BEC only! Need to write entirely new code for other channel formations"""

    def __init__(self, p: float):
        self.p = p

    def transmit_message(self, bits: List[Bit]):
        out_bits = [Bit(b.real) for b in bits]
        for bit in out_bits:
            if random.random() < self.p:
                bit.erase()
        return out_bits



class VariableCheckEdge:

    """
    The edge between a variable and check node in the LDPC bipartite graph
    This stores the most recently passed vc and cv messages, which can be accessed by the nodes & by the visualisation code
    """

    def __init__(self, v_node, c_node, v_idx, c_idx):
        self.cv_message = Bit(np.nan)
        self.vc_message = Bit(np.nan)
        self.v_node = v_node
        self.c_node = c_node
        v_node.add_edge(c_idx, self)
        c_node.add_edge(v_idx, self)

    def __repr__(self):
        return f"|CV:{self.cv_message.__repr__()}|VC:{self.vc_message.__repr__()}|"