import pandas as pd
import pydynaa
import functools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
from netgraph import MultiGraph

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction, QSource, SourceStatus, QuantumChannel, ClassicalChannel
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel, T1T2NoiseModel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_MEASURE_X, INSTR_SWAP, INSTR_CNOT, INSTR_H, INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.nodes import Node, Network, Connection
from netsquid_qswitch.network import ExponentialDelayModel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.state_sampler import StateSampler
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.util import simlog
import logging

'''
logger = logging.getLogger('netsquid')
simlog.logger.setLevel(logging.DEBUG)
# Create a file handler and set the filename
log_file_path = 'simulation.log'
file_handler = logging.FileHandler(log_file_path)

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)'''


SWITCH_NODE_BASENAME = "switch_node_"
END_NODE_BASENAME = "end_node_" 

      
class NetworkGraph:
    def __init__(self, weights):
        self.weights = weights
        self.G = nx.MultiGraph()      
        nodeleft, node, noderight = 'Alice', 'Repeater', 'Bob'
        self.G.add_nodes_from([nodeleft, node, noderight])
        
        self.edge_to_weight = []
        for j in range(len(self.weights)//2):
            edge_ab = (nodeleft, node, j)
            edge_bc = (node, noderight, j)
            weight_ab = round(self.weights[j], 2)
            weight_bc = round(self.weights[j + len(self.weights)//2], 2)
            
            self.edge_to_weight.append((edge_ab, weight_ab))
            self.edge_to_weight.append((edge_bc, weight_bc))
        self.edge_to_weight = dict(self.edge_to_weight)

        self.edge_colors = {}
        self.edge_widths = {}
        for j, edge, weight in zip(range(len(self.edge_to_weight)), self.edge_to_weight, self.edge_to_weight.values()):
            if j == (len(self.edge_to_weight)-1):  # Highlight the best A-B edge in red
                self.edge_colors[edge] = 'red'
                
            elif j == (len(self.edge_to_weight)-2) :  # Highlight the best B-C edge in red
                self.edge_colors[edge] = 'red'
            else:
                self.edge_colors[edge] = 'black'
            self.edge_widths[edge] = 2.5*weight #esto se puede quitar, no se
                
        for edge, weight in self.edge_to_weight.items():
            self.G.add_edge(*edge, weight=weight)
        self.edge_positions={nodeleft: (0,0.5), node: (0.5,0.5), noderight: (1,0.5)} # lo hago asi porque node_layout='linear' no funciona
        
    def visualize_weighted_graph(self):
        MultiGraph(self.G, node_labels=True, edge_labels=self.edge_to_weight, edge_color=self.edge_colors, node_size=5.5,
                   edge_width=self.edge_widths, node_layout=self.edge_positions, node_label_fontdict=dict(size=10), edge_label_fontdict=dict(size=9))

        plt.show()

class FidelityObtainer(LocalProtocol):
    def __init__(self, nodes, name):
        super().__init__(nodes, name)
        self.final_secpos = None
        self.final_firstpos = None
        self.nodelist=nodes

    def PositionDecision(self):
            # Iterate over pairs of neighboring nodes
        node_left, node, node_right = self.nodelist[0], self.nodelist[1], self.nodelist[2]
        # Initialize lists to store pairs for each channel
        pairs_channel1 = []
        positions1= []
        pairs_channel2 = []
        positions2=[]
        n = node.qmemory.num_positions//2
        global final_initialpos, final_endpos
        for pos in range(n):
            #print(pos)
            #print(2*pos + 1)
            pair1 = [qubit[0] for qubit in [node_left.qmemory.peek([2*pos]), node.qmemory.peek([2*pos + 1])]] #Pairs of the first channel
            pair2 = [qubit[0] for qubit in [node.qmemory.peek([2*pos]), node_right.qmemory.peek([2*pos + 1])]] #Pairs of the second channel
            #print(pair1)
            #print(pair2)
            pairs_channel1.append(pair1) #We have to save the pairs but also the memory positions
            positions1.append(2*pos +1)  #to be accessed later by the node
            
            pairs_channel2.append(pair2)
            positions2.append(2*pos)
        
        # Calculate fidelities
        fidelities_channel1 = [ns.qubits.fidelity(pair, ks.b00, squared=True) for pair in pairs_channel1]
        fidelities_channel2 = [ns.qubits.fidelity(pair, ks.b00, squared=True) for pair in pairs_channel2]
        
        weights=sorted(fidelities_channel1)+ sorted(fidelities_channel2)
        #print(weights)
        #graph=NetworkGraph(weights=weights)
        #graph.visualize_weighted_graph()

        # Determine positions to measure
        first_index = fidelities_channel1.index(max(fidelities_channel1))     
        sec_index = fidelities_channel2.index(max(fidelities_channel2))
        
        self.first_pos= positions1[first_index]
        self.sec_pos=positions2[sec_index]

        if 'end_node' in node_left.name:
            final_initialpos = self.first_pos - 1 # The first position in the left node is related like this
        if 'end_node' in node_right.name:
            final_endpos = self.sec_pos + 1# The first position in the right node is related as before
        return self.sec_pos, self.first_pos
    
    
    
    @property
    def FirstPositionBefore(self):
        return self.first_pos - 1
    
    @property
    def SecondPositionAfter(self):
        return self.sec_pos + 1
    
    @property
    def FinalFirstPosition(self):
        return self.final_firstpos
    
    @property
    def FinalSecondPosition(self):
        return self.final_secpos
                
class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name, nodelist, qmempairs=2):
        super().__init__(node, name)
        self.nodelist= nodelist
        self.node=node
        self.qmempairs= qmempairs
        self._qmem_input_port_l = [None] * self.qmempairs
        self._qmem_input_port_r = [None] * self.qmempairs
        for i in range(self.qmempairs):
            self._qmem_input_port_l[i] = self.node.qmemory.ports[f"qin{(2*i)+1}"]
            self._qmem_input_port_r[i] = self.node.qmemory.ports[f"qin{2*i}"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):  
        while True:
            events_l = [self.await_port_input(port_l) for port_l in self._qmem_input_port_l] #await all left input ports
            events_r = [self.await_port_input(port_r) for port_r in self._qmem_input_port_r] #await all right input ports

            combined_event_l = events_l[0] if len(events_l) == 1 else functools.reduce(lambda x, y: x & y, events_l) #combine the events with AND
            combined_event_r = events_r[0] if len(events_r) == 1 else functools.reduce(lambda x, y: x & y, events_r)
            yield (combined_event_l & combined_event_r)
            # Perform Bell measurement
            routing= FidelityObtainer(nodes=self.nodelist, name="Channel_Decider")
            secpos, firstpos = routing.PositionDecision()
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[secpos, firstpos])
            m = [self._program.output["m"], routing.SecondPositionAfter, routing.FirstPositionBefore]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))
            self.node.ports["ccon_R2"].tx_output(Message(m))
            self.node.ports["ccon_L3"].tx_output(Message(m))
            
class SwapProtocol2(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name, qmempairs=2):
        super().__init__(node, name)
        self.qmempairs= qmempairs
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):  
        while True:
            yield (self.await_port_input(self.node.ports["ccon_R3"])& self.await_port_input(self.node.ports["ccon_L2"]))
            # Perform Bell measurement
            message1 = self.node.ports["ccon_L2"].rx_input()
            #print(self.node, message1)
            message2 = self.node.ports["ccon_R3"].rx_input()
            #print(message2)
            secpos, firstpos = message1.items[1], message2.items[2]
            #print(secpos, firstpos)
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[secpos, firstpos])
            m = [self._program.output["m"], message2.items[1], message1.items[2]]
            #print('swap2',m)
            # Send result to right node on end
            self.node.ports["ccon_R4"].tx_output(Message(m))
                
class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()

class CorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """

    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            expression_or=self.await_port_input(self.node.ports["ccon_L"]) | self.await_port_input(self.node.ports["ccon_L4"])
            yield expression_or
            message = []
            if expression_or.first_term.value:
                message.append(self.node.ports["ccon_L"].rx_input().items[0])
            if expression_or.second_term.value:
                message.append(self.node.ports["ccon_L4"].rx_input().items[0])
            if message is None:
                continue
            m, = message[0]
            
            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr += 1
            self._counter += 1
            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[final_endpos])
                self.send_signal(Signals.SUCCESS)               
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0
                
class FibreDepolarizeModel(QuantumErrorModel):
    """Custom non-physical error model used to show the effectiveness
    of repeater chains.

    The default values are chosen to make a nice figure,
    and don't represent any physical system.

    Parameters
    ----------
    p_depol_init : float, optional
        Probability of depolarization on entering a fibre.
        Must be between 0 and 1. Default 0.009
    p_depol_length : float, optional
        Probability of depolarization per km of fibre.
        Must be between 0 and 1. Default 0.025

    """

    def __init__(self):
        super().__init__()
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            dgd=0.6*np.sqrt(float(kwargs['length'])/50)
            tau=gauss(dgd, dgd)
            #print(tau)
            tdec=1.6
            if tau >= tdec:
                prob=1
            elif tau < tdec:
                prob=0
            #print(prob)
            ns.qubits.depolarize(qubit, prob=prob)

def _create_qconnection(connection_number, num_position, distance, single_hop_state,
                        single_hop_timing_model):
    """
    A quantum connection between two consecutive nodes.
    The connection continuously produces entangled pairs of qubits
    and spits out one to each side (internally, this is done by a
    :obj:`netsquid.components.components.qsource.QSource`).

    Parameters
    ----------
    name : str
    distance: float
    single_hop_state : :obj:`netsquid.qubits.qstate.QState`
        The two-qubit state that is produced by the connection.
    single_hop_timing_model :  :obj:`netsquid.components.delaymodel.DelayModel`
        The timing at which the two-qubit states are produced, one after
        another.

    Returns
    -------
    :obj:`netsquid.nodes.connection.Connection`

    Note
    ----
    The ports of the connection are identified as

      * A: to the switch node
      * B: to the leaf node
    """

    # quantum connection
    state_sampler = StateSampler([single_hop_state], [1.0])#StateSampler(qs_reprs=[single_hop_state], probabilities=[1])

    qsource = QSource("qsource{}_pos{}".format(connection_number, num_position),
                      state_sampler=state_sampler,
                      num_ports=2,
                      timing_model=single_hop_timing_model,
                      status=SourceStatus.INTERNAL)

    # We adjust the distances to simulate that the photons are being generated in one of the nodes (the leftmost of each connection)
    qchannel_M2left = QuantumChannel(
        name="qchannel_M2left{}_pos{}".format(connection_number, num_position),
        length=1e-9,
        models={"delay_model": FibreDelayModel(2e5),
                "quantum_noise_model": FibreDepolarizeModel()})
    qchannel_M2right= QuantumChannel(
        name="qchannel_M2right{}_pos{}".format(connection_number, num_position),
        length=distance,
        models={"delay_model": FibreDelayModel(2e5),
                "quantum_noise_model": FibreDepolarizeModel()})

    # classical_connection, for now deprecated
    
    '''cchannel = ClassicalChannel("cchannel2leaf{}".format(connection_number),
                              length=distance)
    connection.add_subcomponent(cchannel)'''
    
    # wrap the two quantum channels and the quantum source into
    # a single connection component
    connection = Connection("qchann{}_pos{}".format(connection_number, num_position))
    connection.add_subcomponent(qsource)
    connection.add_subcomponent(qchannel_M2left)
    connection.add_subcomponent(qchannel_M2right)


    # link the subcomponents internally
    qsource.ports["qout0"].connect(qchannel_M2right.ports["send"])
    qsource.ports["qout1"].connect(qchannel_M2left.ports["send"])
    qchannel_M2left.ports["recv"].forward_output(connection.ports["A"])
    qchannel_M2right.ports["recv"].forward_output(connection.ports["B"])

    return connection

def _create_quantumprocessor(name, num_positions, T1, T2):
    """
    Parameters
    ----------
    name : str
    num_positions : int
    
    Returns
    -------
    :obj:`netsquid.components.qprocessor.QuantumProcessor`
    """
    gate_duration= 0
    qnoise_model= None
    physical_instructions = [
        PhysicalInstruction(INSTR_MEASURE, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_MEASURE_X, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_SWAP, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_CNOT, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_H, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_X, duration=gate_duration, quantum_noise_model=qnoise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration, quantum_noise_model=qnoise_model)
    ]
    qprocessor = QuantumProcessor(name=name,
                                  num_positions=num_positions,
                                  fallback_to_nonphysical=False,
                                  mem_noise_models=[T1T2NoiseModel(T1= T1, T2=T2)] * num_positions,
                                  phys_instructions=physical_instructions)
    return qprocessor

def setup_network(number_of_switches, distances,
                  single_hop_state, single_hop_timing_models,
                  num_positions, T1, T2):
    """
    Constructs a linear topology network with multiple centre nodes (switches) and
    two end nodes.

    Parameters
    ----------
    number_of_switches : int
    distances_from_centre : list of int
    single_hop_state : :obj:`netsquid.qubits.qstate.QState`
    single_hop_timing_models : list of
        :obj:`netsquid.components.models.delaymodels.DelayModel`
    num_positions : int
        Number of qubits in each quantum processor (one for each node)
    T2 : int
    Returns
    -------
    :obj:`netsquid.components.component.Component`
    """
    network = Network("linear_network")

    # Create switch nodes with quantum processors
    switch_nodes = [Node("{}{}".format(SWITCH_NODE_BASENAME, ix),qmemory=_create_quantumprocessor('endnode_qmem{}'.format(ix),
                        num_positions=num_positions, T1=T1, T2=T2)) for ix in range(number_of_switches)]

    # Create end nodes with quantum processors
    end_nodes = [Node("{}{}".format(END_NODE_BASENAME, ix),qmemory=_create_quantumprocessor('endnode_qmem{}'.format(ix),
                        num_positions=num_positions, T1=T1, T2=T2)) for ix in range(2)]

    nodes=[end_nodes[0]] + switch_nodes + [end_nodes[1]] #useful for adding equipment
    
    network.add_nodes(nodes)

    # add classical connections
    for i in range(len(nodes)-1):
        node, node_right = nodes[i], nodes[i + 1]
        cconn1 = ClassicalConnection(name=f"cconn_{i}-{i+1}", length=distances if isinstance(distances,int) else distances[i])
        cconn2 = ClassicalConnection(name=f"cconn_{i}-{i+1}_2", length=distances if isinstance(distances,int) else distances[i])
        cconn3 = ClassicalConnection(name=f"cconn_{i}-{i+1}_3", length=distances if isinstance(distances,int) else distances[i])
        cconn4 = ClassicalConnection(name=f"cconn_{i}-{i+1}_4", length=distances if isinstance(distances,int) else distances[i])
        network.add_connection(
            node, node_right, connection=cconn1, label="classical_corrections",
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        network.add_connection(
            node, node_right, connection=cconn4, label="classical_corrections2",
            port_name_node1="ccon_R4", port_name_node2="ccon_L4")
        network.add_connection(
            node, node_right, connection=cconn2, label="classical_forw_positions",#forward direction
            port_name_node1="ccon_R2", port_name_node2="ccon_L2")
        network.add_connection(
            node_right, node, connection=cconn3, label="classical_back_positions",#backward direction
            port_name_node1="ccon_L3", port_name_node2="ccon_R3")
        
        
    # Forward cconn to right most node
    if "ccon_L" in node.ports:
        node.ports["ccon_L"].bind_input_handler(
            lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))
        node.ports["ccon_L4"].bind_input_handler(
            lambda message, _node=node: _node.ports["ccon_R4"].tx_output(message))
    
    if isinstance(distances,int): #if the distance is fixed, convert to a list of n distances
        distances=[distances]*len(nodes)

    # add quantum connections between nodes
    for i in range(len(nodes)-1):
        node , node_right=nodes[i], nodes[i+1]
        for j in range(num_positions//2):
            qconnection = _create_qconnection(
                connection_number=i,
                num_position=j,
                distance=distances[i],
                single_hop_state=single_hop_state,
                single_hop_timing_model=single_hop_timing_models[i])
            port_name, port_r_name= network.add_connection(node, node_right, connection=qconnection,
                                                           label=f"quantum_{node.name}_{node_right.name}_{j}")
            # Connect all memory positions to its respective ports
            node.ports[port_name].forward_input(node.qmemory.ports[f"qin{2*j}"])
            node_right.ports[port_r_name].forward_input(node_right.qmemory.ports[f"qin{2*j+1}"]) 

    return network

def setup_repeater_protocol(network,mem_pairs):
    """Setup repeater protocol on repeater chain network.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    Returns
    -------
    :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    """
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    end_nodes = [node for node in network.nodes.values() if END_NODE_BASENAME in node.name]
    switch_nodes = [node for node in network.nodes.values() if SWITCH_NODE_BASENAME in node.name]
    nodes=[end_nodes[0]] + switch_nodes + [end_nodes[1]]
    if len(nodes)==3:
        subprotocol1 = SwapProtocol(node=nodes[1], name=f"Swap_{nodes[1].name}", nodelist=nodes,qmempairs=mem_pairs)
        protocol.add_subprotocol(subprotocol1)
    else:
        for i, node in zip(range(1, len(nodes), 2), nodes[1:-1:2]):
            #print(i, node)
            subprotocol1 = SwapProtocol(node=node, name=f"Swap_{node.name}", nodelist=[nodes[i-1], nodes[i], nodes[i+1]],qmempairs=mem_pairs)
            protocol.add_subprotocol(subprotocol1)
            
        for node in nodes[2:-2:2]:
            #print(node)
            subprotocol2 = SwapProtocol2(node=node, name=f"Swap2_{node.name}",qmempairs=mem_pairs)
            protocol.add_subprotocol(subprotocol2)

    # Add CorrectProtocol to Bob
    subprotocol3 = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(subprotocol3)
    return protocol

def setup_datacollector(network, protocol):
    """Setup the datacollector to calculate the fidelity
    when the CorrectionProtocol has finished.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    protocol : :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    Returns
    -------
    :class:`~netsquid.util.datacollector.DataCollector`
        Datacollector recording fidelity data.

    """
    # Ensure nodes are ordered in the chain:
    end_nodes = [node for node in network.nodes.values() if END_NODE_BASENAME in node.name]
    switch_nodes = [node for node in network.nodes.values() if SWITCH_NODE_BASENAME in node.name]
    nodes=[end_nodes[0]] + switch_nodes + [end_nodes[1]]

    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([final_initialpos])
        qubit_b, = nodes[-1].qmemory.peek([final_endpos])
        #print(final_endpos,final_initialpos)
        fidelity1 =ns.qubits.fidelity([qubit_b, qubit_a], ks.b00, squared=True)
        return {"fidelity": fidelity1}#,"fidelity 2": fidelity2,"Highest fidelity ch1":fidelityc1, "Highest fidelity ch2":fidelityc2}

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    #print(protocol.subprotocols)
    dc.collect_on([pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value)])
    return dc

def run_simulation(num_nodes=3, node_distances=500, num_iters=5, mem_positions=4):
    """Run the simulation experiment and return the collected data.

    Parameters
    ----------
    num_nodes : int, optional
        Number nodes in the repeater chain network. At least 3. Default 4.
    node_distance : float, optional
        Distance between nodes, larger than 0. Default 20 [km].
    num_iters : int, optional
        Number of simulation runs. Default 100.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with recorded fidelity data.

    """
    ns.sim_reset()
    est_runtime = (0.5 + num_nodes - 1) * node_distances * 5e3
    network = setup_network(number_of_switches= num_nodes - 2, distances=node_distances,
                  single_hop_state=ks.b00, single_hop_timing_models=[FixedDelayModel(delay=1e9)]*(num_nodes -1),
                  num_positions=mem_positions , T1=0e-6, T2=0.0e-6)
    protocol = setup_repeater_protocol(network, mem_pairs=mem_positions//2)
    dc = setup_datacollector(network, protocol)
    protocol.start()
    #ns.sim_run()
    ns.sim_run(50*est_runtime*num_iters)
    #print(dc.dataframe)

    print(f" Average fidelity of {dc.dataframe['fidelity'].mean():.3f} with a STD of {dc.dataframe['fidelity'].std():.3f}")
    #print(f"Channels selected: {dc.dataframe['channel 1'].mean():.2f} and {dc.dataframe['channel 2'].mean():.2f} ")
    #print(f" Average fidelity 2 of {dc.dataframe['fidelity 2'].mean():.3f} with a STD of {dc.dataframe['fidelity 2'].std():.3f}")
    return dc.dataframe


#def create_plot(num_iters=2000):
    """Run the simulation for multiple nodes and distances and show them in a figure.

    Parameters
    ----------

    num_iters : int, optional
        Number of iterations per simulation configuration.
        At least 1. Default 2000.
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for distance in [10, 30, 50]:
        data = pandas.DataFrame()
        for num_node in range(3, 20):
            data[num_node] = run_simulation(num_nodes=num_node,
                                            node_distance=distance / num_node,
                                            num_iters=num_iters)['fidelity']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        data.plot(y='fidelity', yerr='sem', label=f"{distance} km", ax=ax)
    plt.xlabel("number of nodes")
    plt.ylabel("fidelity")
    plt.title("Repeater chain with different total lengths")
    plt.show()

def create_plot(num_iters=2):

    fig, ax = plt.subplots()
    fidelities=[]
    errors=[]
    for mempos in range(2,21,2):
        data = pd.DataFrame()
        for distance in np.linspace(500, 501,50):
            result = run_simulation(num_nodes=3,
                                    node_distances=int(distance),
                                    num_iters=num_iters,mem_positions=mempos)['fidelity']
            data[distance] = result
        #print(data)
    # Calculate mean and sem outside the loop
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        #data.plot(y='fidelity', label=f"{mempos} positions", ax=ax)
        fidelities.append(data['fidelity'].mean())
        errors.append(data['fidelity'].std())
        #print(f" Average fidelity of {data['fidelity'].mean():.3f} with a STD of {data['fidelity'].std():.3f}")
    #print(fidelities)
    plt.errorbar(range(2,21,2),fidelities)
    plt.xlabel("Number of memories")
    plt.xticks(range(2,21,2))
    plt.ylabel("Fidelity")
    plt.ylim(0.4, 1)
    plt.title("2-hop system for different memory positions")
    plt.show()

ns.set_qstate_formalism(ns.QFormalism.DM)
#random_state = np.random.RandomState(42)
#set_random_state(rng=random_state)

#run_simulation()

create_plot(2)