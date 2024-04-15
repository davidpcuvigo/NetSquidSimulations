'''
ENTANGLEMENT SWAPPING ROUTING FOR 2 MEMORY POSITIONS

We use NetSquid to simulate a quantum network of N nodes (N has to be odd. Solution is WIP). An example network could be like this, for N=5:
                            +-----------+   +----------+   +-----------+   +----------+   +-------------+ 
                            |   Alice   |   |          |   |           |   |          |   |     Bob     |
                            |   "End    0---1 "Switch  0---1 "Switch   0---1 "Switch  0---1    "End     |
                            |  Node 0"  |   |  Node 0" |   |  Node 1"  |   |  Node 2" |   |    Node 1"  |
                            |           2---3          2---3           2---3          2---3             |
                            |           |   |          |   |           |   |          |   |             |
                            +-----------+   +----------+   +-----------+   +----------+   +-------------+
                                            SwapProtocol   SwapProtocol2   SwapProtocol   CorrectProtocol
                                            
                                            
We will refer to the outer nodes as *end nodes*, and sometimes also as Alice and Bob for convenience,
and the in between nodes as the *switch nodes*.
The lines between the nodes represent an entangling connection between memory positions (with its associated number), 
while classical channels have been ommitted.

The switch nodes will use, depending on the parity of their position in the network, one of the two swap Protocols. SwapProtocol, consists
of the following steps:

1. generating entanglement with both of its neighbours,
2. finding the best possible fidelity in its memory qubits,
3. measuring the two locally stored qubits in the Bell basis,
4. sending backwards and forward the memory positions in which to measure to its nearest neighbours (info used in SwapProtocol2),
5. sending its own measurement outcomes to its right neighbour, and also forwarding on outcomes received from its left neighbour in this way.

Now, for SwapProtocol2:

1. generating entanglement with both of its neighbours,
2. waiting to receive from nearest neighbours in which positions the measurement has to be made,
3. measuring the two locally stored qubits in the Bell basis,
4. sending its own measurement outcomes to its right neighbour, and also forwarding on outcomes received from its left neighbour in this way.

The definition of the correction subprotocol responsible for applying the classical corrections at Bob is CorrectProtocol:

1. wait for classical inputs,
2. read the classical message and collect all the correction gates from every switch node,
3. apply the final correction in the proper position. 


Note that for the corrections we only need to apply each of the X or Z operators maximally once.
This is due to the anti-commutativity and self-inverse of these Pauli matrices, i.e.
ZX = -XZ and XX=ZZ=I, which allows us to cancel repeated occurrences up to a global phase (-1).
The program that executes the correction on Bob's quantum processor is SwapCorrectProgram.

- create_qprocessor:

We use this function to create quantum processors for each node, which can all have different specifications.

- setup_network:

We create a network component and add the nodes and connections to it. This way we can easily keep track of all our components
in the network, which will be useful when collecting data later. We have used a custom noise model in this example, which helps
to exaggerate the effectiveness of the repeater chain, called FibreDepolarizeModel.

- setup repeater_protocol:

To easily manage all the protocols, we add them as subprotocols of one main protocol.
In this way, we can start them at the same time, and the main protocol will stop when all subprotocols have finished.

- setup_datacollector:

With our network and protocols ready, we can add a data collector to define when and which data we want to collect.
We can wait for the signal sent by the *CorrectProtocol* when it finishes, and if it has, compute the fidelity of the
qubits at Alice and Bob with respect to the expected Bell state.
Using our network and main protocol we can easily find Alice and the CorrectionProtocol as subcomponents.

- run_simulation:

We aggregate all the components together and run the simulation for the number of iterations we desire. It works with simulation
time, rather than with number of iterations, so this number of iterations is approximate.

WIP:
- Implement the use of M memory positions: as long as there is not a quantum switch, we have to create an Entangling Connection for
every memory pair. The protocols are already adapted to M positions and ports, but the physical layer in setup_network is not.

- Change the last protocol for even number of switch nodes: SwapProtocol2 receives information from left and right, so when its rightmost
node is Bob, the protocol fails. We have to implement a protocol that works like SwapProtocol but obtains the fidelity only from
its right neighbour, and not its left neighbour.

- Expand the network to not only linear schemes, so the switch nodes actually act like switches and not as repeaters. The classes used 
in the Quantum Switch snippet will be of great help.

- The unused NetworkGraph class has the objective of representing the fidelity of each channel individually and the final fidelity between 
Alice and Bob. The work is mostly done, but needs finishing.

'''


import pandas
import pydynaa
import functools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from random import uniform, gauss
import re
from configparser import ConfigParser
import ast
from statistics import mean, median

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction, ClassicalChannel
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z, INSTR_I, INSTR_SWAP
from netsquid.nodes import Node, Network
from netsquid.nodes.connections import DirectConnection
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.util.datacollector import DataCollector
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
#from netsquid.examples.purify import Distil, Filter
from netsquid.util import simlog
from netsquid.util.constrainedmap import ConstrainedMapView, ValueConstraint, ConstrainedMap
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
#from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.component import Message, Port
import netsquid.qubits.operators as ops
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from pydynaa import EventExpression

import logging

fidelities = []
'''logger = logging.getLogger('netsquid')
simlog.logger.setLevel(logging.DEBUG)
# Create a file handler and set the filename
log_file_path = 'simulation.log'
file_handler = logging.FileHandler(log_file_path)

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)'''


def readProperties(conf,file):
    '''
    Reads each property from the config file
    Input: 
        conf: dictionary where key-values will be mapped
        file: path and name of properties file
    Output: by reference in dictionary
    '''
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(file)

    floatExp = re.compile('^[-+]?[0-9]+\.[0-9]+$')
    sciExp = re.compile('[+\-]?[^A-Za-z]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)')
    intExp = re.compile('^[-+]?[0-9]+$')
    listExp = re.compile('^\[(\d+,\s*)+\d+\]$')

    #Find each property in each of the sections and cast as integer or float
    for section in config.sections():
        for key in config[section]:
            if floatExp.match(config[section][key]) != None or sciExp.match(config[section][key]) != None:
                conf[key] = float(config[section][key])
            elif intExp.match(config[section][key]):
                conf[key] = int(config[section][key])
            elif listExp.match(config[section][key]):
                conf[key] = ast.literal_eval(config[section][key])
            else:
                raise ValueError('Only integers and floats are allowed as properties, invalid value in {}'.format(key))

    '''To be used once distances list is used
    if isinstance(conf['node_distances'],int): #if the distance is fixed, convert to a list of n distances
        conf['node_distances']=[conf['node_distances']]*(conf['num_nodes']-1)
    
    if len(conf['node_distances']) != conf['num_nodes']-1:
        raise ValueError('Number of nodes does not match number of elements in node_distances')
    '''

SWITCH_NODE_BASENAME = "switch_node_"
END_NODE_BASENAME = "end_node_" 

class Filter(NodeProtocol):
    """Protocol that does local filtering on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event expression node should wait for before starting filter.
        This event expression should have a
        :class:`~netsquid.protocols.protocol.Protocol` as source and should by fired
        by signalling a signal by this protocol, with the position of the qubit on the
        quantum memory as signal result.
        Must be set before the protocol can start
    msg_header : str, optional
        Value of header meta field used for classical communication.
    epsilon : float, optional
        Parameter used in filter's measurement operator.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    Attributes
    ----------
    meas_ops : list
        Measurement operators to use for filter general measurement.

    """

    def __init__(self, node, port, start_expression=None, msg_header="filter",
                 epsilon=0.3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "Filter({}, {})".format(node.name, port.name)
        super().__init__(node, name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_OK = False
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_measurement_operators(epsilon)

    def _set_measurement_operators(self, epsilon):
        m0 = ops.Operator("M0", np.sqrt(epsilon) * outerprod(s0) + outerprod(s1))
        m1 = ops.Operator("M1", np.sqrt(1 - epsilon) * outerprod(s0))
        self.meas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                self._qmem_pos = ready_signal.result
                yield from self._handle_qubit_rx()

    # TODO does start reset vars?
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_OK = False
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.get_position_used(self._qmem_pos):
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        # Handle incoming Qubit on this node.
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # Retrieve Qubit from input store
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        m = output["instr"][0]
        #m = INSTR_MEASURE(self.node.qmemory, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        self.local_qcount += 1
        self.local_meas_OK = (m == 0)
        self.port.tx_output(Message([self.local_qcount, self.local_meas_OK], header=self.header))
        self._check_success()

    def _handle_cchannel_rx(self):
        # Handle incoming classical message from sister node.
        if (self.local_qcount == self.remote_qcount and
                self._qmem_pos is not None and
                self.node.qmemory.get_position_used(self._qmem_pos)):
            self._check_success()

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            # SUCCESS!
            self.send_signal(Signals.SUCCESS, self._qmem_pos)
            #print('JUANCHECKSUCCESS nodo {} posicion {}'.format(self.node,self._qmem_pos))
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status
            pass
        else:
            # FAILURE
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)

    def _handle_fail(self):
        if self.node.qmemory.get_position_used(self._qmem_pos):
            self.node.qmemory.pop(positions=[self._qmem_pos])

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 1:
            return False
        return True

class Distil(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        if role.upper() == 'A':
            self.port2 = self.node.ports["purification_port_correct"]
        elif role.upper() == 'B':
            self.port2 = self.node.ports["purification_port_correct2"]
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        self.role = role
        self.temp = 6
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog

    def run(self):

        while True:
            cchannel_ready = self.await_port_input(self.port) 
            qmemory_ready = self.await_port_input(self.port2) 
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    position = 0 if self.role.upper() == 'A' else 1
                    yield from self._handle_new_qubit(position)
            elif expr.second_term.value:
                classical_message, = self.port2.rx_input().items
                yield from self._handle_new_qubit(classical_message)
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
        # Process signalling of new entangled qubit
        self._waiting_on_second_qubit = True
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            position_check= 0 if self.role.upper() == 'A' else 1
            assert not self.node.qmemory.mem_positions[position_check].is_empty
            assert memory_position != self._qmem_positions[0]
            self.local_qcount += 1
            self._qmem_positions[1] = memory_position
            self._qmem_positions[0] = 2 - memory_position if self.role.upper() == 'A' else 4 - memory_position
            #self._waiting_on_second_qubit = False
            
            yield from self._node_do_DEJMPS()
        else:
            # New candidate for first qubit arrived
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2])  # If instruction not instant
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[1] = None
        # Send local results to the remote node to allow it to check for success.
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, self._qmem_positions[0])
            else:
                try:
                # FAILURE
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.node.ports[self.port2].tx_output(Message(self.local_qcount))
                except:
                    ns.sim_stop()
                    if len(fidelities) < 100:
                        ns.sim_reset()
                        create_plot1()
                    else:
                        file = open('fidelities.txt','w')
                        for item in fidelities:
                            file.write(str(item)+"\n")
                        file.close()
                        print(mean(fidelities),len(fidelities))
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True

class EntangleNodes(NodeProtocol):
    """Cooperate with another node to generate shared entanglement.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node`
        Node to run this protocol on.
    role : "source" or "receiver"
        Whether this protocol should act as a source or a receiver. Both are needed.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event Expression to wait for before starting entanglement round.
    input_mem_pos : int, optional
        Index of quantum memory position to expect incoming qubits on. Default is 0.
    num_pairs : int, optional
        Number of entanglement pairs to create per round. If more than one, the extra qubits
        will be stored on available memory positions.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None, qsource=None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError
        self._is_source = role.lower() == "source"
        name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
        self._qsourcereal = qsource

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
                                   .format(self._num_pairs - 1, len(unused_positions)))
            #for i in unused_positions[:extra_memory]:
            for i in unused_positions[::]:
                counter=0
                if (i+self._input_mem_pos)%2 == 0:
                    counter += 1
                    self._mem_positions.append(i)
                    self.node.qmemory.mem_positions[i].in_use = True
                if counter == extra_memory:
                    break
        # Call parent start method
        return super().start()
        

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):
        self._qsource_name = self._qsourcereal
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self._is_source and self.entangled_pairs >= self._num_pairs:
                # If no start expression specified then limit generation to one round
                break
            for mem_pos in self._mem_positions[::-1]:
                # Iterate in reverse so that input_mem_pos is handled last
                if self._is_source:
                    self.node.subcomponents[self._qsource_name].trigger()
                yield self.await_port_input(self._qmem_input_port)
                if mem_pos != self._input_mem_pos:                   
                    self.node.qmemory.execute_instruction(
                        INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                self.entangled_pairs += 1
            self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True

class NetworkGraph:
    def __init__(self, network):
        self.network = network
        self.nodelist = network
        self.num_nodes = len(self.nodelist)
        self.graph = self.create_weighted_graph()

    def create_weighted_graph(self):
        G=nx.Graph()
        n = self.num_nodes       
        for i in range(n-1):
            node, nodeleft= self.nodelist[i], self.nodelist[i+1]
            for pos in range(n//2):
                pair = [qubit[0] for qubit in [node.qmemory.peek([2*pos]), nodeleft.qmemory.peek([2*pos + 1])]]
                auxfidelity = [ns.qubits.fidelity(pair, ks.b00, squared=True)]
                fidelity = auxfidelity[0]
                node1, node2, weight = self.nodelist[i], self.nodelist[i+1], fidelity
                G.add_edge(node1, node2, weight= weight)        
        return G
    
    def visualize_weighted_graph(self):
        pos = nx.spring_layout(self.graph)  # Positions for all nodes
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

class FidelityObtainer(LocalProtocol):
    def __init__(self, nodes, name, fidelity_threshold=0.8):
        super().__init__(nodes, name)
        self.final_secpos = None
        self.final_firstpos = None
        self.nodelist=nodes
        self.fidelity_threshold=fidelity_threshold


    def PositionDecision(self):
            # Iterate over pairs of neighboring nodes
        node_left, node, node_right = self.nodelist[0], self.nodelist[1], self.nodelist[2]
        # Initialize lists to store pairs for each channel
        pairs_channel1 = []
        positions1= []
        pairs_channel2 = []
        positions2=[]
        n = node.qmemory.num_positions//2 -1

        for pos in range(n):
            pair1 = [qubit[0] for qubit in [node_left.qmemory.peek([2*pos]), node.qmemory.peek([2*pos + 1])]] #Pairs of the first channel
            pair2 = [qubit[0] for qubit in [node.qmemory.peek([2*pos]), node_right.qmemory.peek([2*pos + 1])]] #Pairs of the second channel
            pairs_channel1.append(pair1) #We have to save the pairs but also the memory positions
            positions1.append(2*pos +1)  #to be accessed later by the node
            
            pairs_channel2.append(pair2)
            positions2.append(2*pos)
        
        # Calculate fidelities
        fidelities_channel1 = [ns.qubits.fidelity(pair, ks.b00, squared=True) for pair in pairs_channel1]
        fidelities_channel2 = [ns.qubits.fidelity(pair, ks.b00, squared=True) for pair in pairs_channel2]
            
        # Determine positions to measure
        first_index = fidelities_channel1.index(max(fidelities_channel1))     
        sec_index = fidelities_channel2.index(max(fidelities_channel2))
        
        self.first_pos= positions1[first_index]
        self.sec_pos=positions2[sec_index]
        global final_firstpos, final_secpos
        if END_NODE_BASENAME in node_left.name:
            final_firstpos = self.first_pos - 1 # The first position in the left node is related like this
        if END_NODE_BASENAME in node_right.name:
            final_secpos = self.sec_pos + 1# The first position in the right node is related as before
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
    
class FidelityObtainerHalf(LocalProtocol):
    def __init__(self, nodes, name):
        super().__init__(nodes, name)
        self.final_secpos = None
        self.final_firstpos = None
        self.nodelist=nodes

    def PositionDecisionHalf(self):
            # Iterate over pairs of neighboring nodes
        node, node_right = self.nodelist[1], self.nodelist[2]
        # Initialize lists to store pairs for each channel
        pairs_channel2 = []
        positions2=[]
        n = node.qmemory.num_positions//2

        for pos in range(n):
            pair2 = [qubit[0] for qubit in [node.qmemory.peek([2*pos]), node_right.qmemory.peek([2*pos + 1])]] #Pairs of the second channel
            pairs_channel2.append(pair2)
            positions2.append(2*pos)
        
        # Calculate fidelities
        fidelities_channel2 = [ns.qubits.fidelity(pair, ks.b00, squared=True) for pair in pairs_channel2]
        
        # Determine positions to measure   
        sec_index = fidelities_channel2.index(max(fidelities_channel2))
        
        self.sec_pos=positions2[sec_index]
        
        if 'end_node' in node_right.name:
            self.final_secpos = self.sec_pos + 1# The first position in the right node is related as before
        return self.sec_pos
    
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

    def __init__(self, node, name, nodelist, purify, qmempairs=2, start_expression=None):
        super().__init__(node, name)
        self.nodelist= nodelist
        self.node=node
        self.purify=purify.lower()
        self.qmempairs= qmempairs
        self._qmem_input_port_l = [None] * self.qmempairs
        self._qmem_input_port_r = [None] * self.qmempairs
        for i in range(self.qmempairs):
            self._qmem_input_port_l[i] = self.node.qmemory.ports[f"qin{(2*i)+1}"]
            self._qmem_input_port_r[i] = self.node.qmemory.ports[f"qin{2*i}"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)
        self.start_expression = start_expression

    def run(self):
        if SWITCH_NODE_BASENAME in self.nodelist[-1].name:
            while True:
                yield(self.start_expression)
                #ic('SALTA SWAP switch',self.node)
                # Perform Bell measurement
                for firstpos in self.qmempairs:
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[2*firstpos + 1, 2*firstpos])
                    m = [self._program.output["m"], 2*firstpos + 1, 1]
                    #ic(m,self.node)
                    # Send result to right node on end
                    self.node.ports["ccon_R"].tx_output(Message(m, header='swaps'))
                    self.node.ports["ccon_R2"].tx_output(Message(m, header='swaps'))
                    self.node.ports["ccon_L3"].tx_output(Message(m, header='swaps'))
                                      
        if END_NODE_BASENAME in self.nodelist[-1].name:
            while True:
                yield(self.start_expression)
                # Perform Bell measurement
                m=[]
                for firstpos in range(self.qmempairs):
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[2*firstpos + 1, 2*firstpos])
                    m.append([self._program.output["m"], 2*firstpos + 1, 1])
                    # Send result to right node on end
                self.node.ports["ccon_R"].tx_output(Message(m, header='swaps'))
                self.node.ports["ccon_R2"].tx_output(Message(m, header='swaps'))
                self.node.ports["ccon_L3"].tx_output(Message(m, header='swaps'))

class SwapProtocol2(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name, nodelist, qmempairs=2, protocol=None):
        super().__init__(node, name)
        self.nodelist=nodelist
        self.qmempairs= qmempairs
        self._qmem_input_port_l = [None] * self.qmempairs
        self._qmem_input_port_r = [None] * self.qmempairs
        for i in range(self.qmempairs):
            self._qmem_input_port_l[i] = self.node.qmemory.ports[f"qin{(2*i)+1}"]
            self._qmem_input_port_r[i] = self.node.qmemory.ports[f"qin{2*i}"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)
        self.protocol = protocol
        
    def run(self): 
        if SWITCH_NODE_BASENAME in self.nodelist[-1].name:
            while True:
                evt_expr_wait = ((self.await_port_input(self.node.ports["ccon_R3"])) | self.await_port_input(self.node.ports["ccon_L2"]))
                yield evt_expr_wait
                #ic('Recibido clásico SW2')
                if evt_expr_wait.first_term.value and evt_expr_wait.second_term.value:
                    # Perform Bell measurement
                    message1 = self.node.ports["ccon_R3"].rx_input()
                    message2 = self.node.ports["ccon_L2"].rx_input()
                    #ic('ambos clásicos')
                elif evt_expr_wait.first_term.value:
                    message1 = self.node.ports["ccon_R3"].rx_input()
                    #ic('primero',message1)
                    yield self.await_port_input(self.node.ports["ccon_L2"])
                    message2 = self.node.ports["ccon_L2"].rx_input()
                else:
                    message2 = self.node.ports["ccon_L2"].rx_input()
                    #ic('segundo',message2)
                    yield (self.await_port_input(self.node.ports["ccon_R3"]))
                    message1 = self.node.ports["ccon_R3"].rx_input()
                secpos, firstpos = message1.items[1], message2.items[2]
                yield self.node.qmemory.execute_program(self._program, qubit_mapping=[secpos, firstpos])
                m = [self._program.output["m"], message2.items[1], message1.items[2]]
                #ic(m)
                # Send result to right node on end
                self.node.ports["ccon_R4"].tx_output(Message(m))

        '''
        if END_NODE_BASENAME in self.nodelist[-1].name:
            while True:
                ic('No debería entrar')
                events_l = [self.await_port_input(port_l) for port_l in self._qmem_input_port_l] #await all left input ports
                events_r = [self.await_port_input(port_r) for port_r in self._qmem_input_port_r] #await all right input ports
                combined_event_l = events_l[0] if len(events_l) == 1 else functools.reduce(lambda x, y: x & y, events_l) #combine the events with AND
                combined_event_r = events_r[0] if len(events_r) == 1 else functools.reduce(lambda x, y: x & y, events_r)
                yield (combined_event_l & combined_event_r & self.await_port_input(self.node.ports["ccon_L2"]))
                # Perform Bell measurement
                routing_last= FidelityObtainerHalf(nodes=self.nodelist, name="Channel_Decider_Last")
                message1 = self.node.ports["ccon_L2"].rx_input()
                secpos, firstpos = message1.items[1], routing_last.PositionDecisionHalf()
                yield self.node.qmemory.execute_program(self._program, qubit_mapping=[secpos, firstpos])
                m = [self._program.output["m"], routing_last.FinalSecondPosition, 'last']
                # Send result to right node on end
                self.node.ports["ccon_R4"].tx_output(Message(m))
        '''           

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

    def __init__(self, node, num_nodes, purify, protocol_order):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self.purify = purify
        self.protocol_order = protocol_order
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0
        self.successes=[False,False] if self.protocol_order == 'SP' else [False]

    def run(self):
        while True:
            expression_or=self.await_port_input(self.node.ports["ccon_L"]) | self.await_port_input(self.node.ports["ccon_L4"])
            
            yield expression_or
            message = []
            if expression_or.first_term.value:
                message.append(self.node.ports["ccon_L"].rx_input().items)
            if expression_or.second_term.value:
                message.append(self.node.ports["ccon_L4"].rx_input().items)
            if message is None:
                continue
            for i in range(2):
                m, = message[0][i][0]
                position=message[0][i][1]
                if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                    self._x_corr += 1
                if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                    self._z_corr += 1
                self._counter += 1
                if self._counter == self.num_nodes - 2:
                    position_a=position if self.purify == 'distil' else final_secpos
                    if self._x_corr or self._z_corr:
                        self._program.set_corrections(self._x_corr, self._z_corr)
                        yield self.node.qmemory.execute_program(self._program, qubit_mapping=[position_a])
                    self.successes[int((position_a-1)/2)] = True             
                    self._x_corr = 0
                    self._z_corr = 0
                    self._counter = 0
            if all(self.successes):
                    self.send_signal(Signals.SUCCESS, position_a-1)
                    self.node.ports["purification_port_correct2"].tx_output(Message(position_a-1))
                                  
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

    def __init__(self, p_depol_init=0.01, p_depol_length=0.01):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
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
            tau=gauss(dgd,dgd)
            tdec=1.6
            if tau >= tdec:
                prob=1
            elif tau < tdec:
                prob=0
            #prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
            #    10, - kwargs['length'] * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)

def create_qprocessor(name,num_positions):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has two memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    noise_rate = conf['noise_rate']
    gate_duration = conf['gate_duration']
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=num_positions, fallback_to_nonphysical=True,
                             mem_noise_models=[mem_noise_model] * num_positions,
                             phys_instructions=physical_instructions)
    return qproc

def setup_network(num_nodes, node_distance, source_frequency):
    """Setup repeater chain network.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network, at least 3.
    node_distance : float
        Distance between nodes [km].
    source_frequency : float
        Frequency at which the sources create entangled qubits [Hz].

    Returns
    -------
    :class:`~netsquid.nodes.network.Network`
        Network component with all nodes and connections as subcomponents.

    """
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    network = Network("Repeater_chain_network")
    # Create nodes with quantum processors
    # Create switch nodes
    number_of_switches=num_nodes - 2
    switch_nodes = [Node("{}{}".format(SWITCH_NODE_BASENAME, ix),qmemory=create_qprocessor('switch_qmem{}'.format(ix), conf['mem_positions']))
                    for ix in range(number_of_switches)]  # Change to the desired number of switches

    # Create end nodes
    end_nodes = [Node("{}{}".format(END_NODE_BASENAME, ix),qmemory=create_qprocessor('endnode_qmem{}'.format(ix),conf['mem_positions']))
                  for ix in range(2)]

    nodes=[end_nodes[0]] + switch_nodes + [end_nodes[1]]#useful for adding equipment
    network.add_nodes(nodes)

    source_fidelity_sq = conf['source_fidelity_sq']
    source_delay = conf['source_delay']

    # Create quantum and classical connections:
    for i in range(num_nodes - 1):
        node, node_right = nodes[i], nodes[i + 1]

        state_sampler = StateSampler([ks.b00, ks.s00],
            probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])

        #QSource
        for pos in range(conf['mem_positions']//2):
            source_node = QSource(
                'qsource_{}{}'.format(i,pos), state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
                models={"emission_delay_model": FixedDelayModel(delay=source_delay)}, label='qsource_{}{}'.format(i,pos))
            node.add_subcomponent(source_node)

            # Setup quantum channels
            qchannel_ar = QuantumChannel(
                'QChannel_{}->{}_{}'.format(i,i+1,pos), length=node_distance,
                models={"quantum_loss_model": FibreDepolarizeModel(), "delay_model": FibreDelayModel(c=200e3)})
            port_name_a, port_name_ra = network.add_connection(
                node, node_right, channel_to=qchannel_ar, label="quantum_{}_{}".format(i,pos))
        #qchannel_br = QuantumChannel(
        #    "QChannel_{}->{}.format(i+1,i)", length=node_distance,
        #    models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=200e3)})
        #port_name_b, port_name_rb = network.add_connection(
        #    node_right, node, channel_to=qchannel_br, label="quantum")

            # Setup node ports:
            node.subcomponents['qsource_{}{}'.format(i,pos)].ports["qout1"].forward_output(
                node.ports[port_name_a])
            node.subcomponents['qsource_{}{}'.format(i,pos)].ports["qout0"].connect(
                node.qmemory.ports["qin{}".format(2*pos)])

            # Send pair to right node
            node_right.ports[port_name_ra].forward_input(node_right.qmemory.ports["qin{}".format(2*pos+1)])
        # Create classical connection
        cconn1 = ClassicalChannel(name=f"cconn_{i}-{i+1}", length=node_distance,transmit_empty_items=False)
        cconn2 = ClassicalChannel(name=f"cconn_{i}-{i+1}_2", length=node_distance,transmit_empty_items=False)
        cconn3 = ClassicalChannel(name=f"cconn_{i}-{i+1}_3", length=node_distance,transmit_empty_items=False)
        cconn4 = ClassicalChannel(name=f"cconn_{i}-{i+1}_4", length=node_distance,transmit_empty_items=False)

        network.add_connection(
            node, node_right, channel_to=cconn1, label="classical_corrections_{}".format(i),
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        network.add_connection(
            node, node_right, channel_to=cconn4, label="classical_corrections2_{}".format(i),
            port_name_node1="ccon_R4", port_name_node2="ccon_L4")
        network.add_connection(
            node, node_right, channel_to=cconn2, label="classical_forw_positions_{}".format(i),#forward direction SP -> SP2
            port_name_node1="ccon_R2", port_name_node2="ccon_L2")
        network.add_connection(
            node_right, node, channel_to=cconn3, label="classical_back_positions_{}".format(i),#backward direction SP -> SP2
            port_name_node1="ccon_L3", port_name_node2="ccon_R3")
        
        #Conexiones de purificación
        for pos in range(conf['mem_positions']//2):
            purification_conn= DirectConnection(name=f'purconn_{i}-{i+1}_{pos}',
                        channel_AtoB=ClassicalChannel(name='AtoB',length=node_distance),
                        channel_BtoA=ClassicalChannel(name='BtoA',length=node_distance)
            )
            network.add_connection(
                node_right, node, connection=purification_conn, label="purification_{}_{}".format(i,pos),#dedicated channel for purification purposes
                port_name_node1="purification_port_A_{}_{}".format(i,pos), port_name_node2="purification_port_B_{}_{}".format(i,pos))

        # Forward cconn to right most node
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(
                lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))
            node.ports["ccon_L4"].bind_input_handler(
                lambda message, _node=node: _node.ports["ccon_R4"].tx_output(message))
            #node.ports["ccon_L"].forward_input(node.ports["ccon_R"])

    for pos in range(conf['mem_positions']//2):
        purification_conn= DirectConnection(name=f'purconn_{i}-{i+1}_{pos}',
                        channel_AtoB=ClassicalChannel(name='AtoB',length=node_distance*(len(nodes)-1)),
                        channel_BtoA=ClassicalChannel(name='BtoA',length=node_distance*(len(nodes)-1))
            )
        network.add_connection(
                end_nodes[0], end_nodes[1], connection=purification_conn, label="purification_{}_{}_SP".format(i,pos),#dedicated channel for purification purposes
                port_name_node1="purification_port_A_0_{}_SP".format(pos), port_name_node2="purification_port_B_0_{}_SP".format(pos))

    purification_conn1= DirectConnection(name=f'purconn_correct',
                        channel_AtoB=ClassicalChannel(name='AtoB',length=node_distance*(len(nodes)-1)),
                        channel_BtoA=ClassicalChannel(name='BtoA',length=node_distance*(len(nodes)-1))
            )
    network.add_connection(
                end_nodes[0], end_nodes[1], connection=purification_conn1, label="purification_SP_correct",#dedicated channel for purification purposes
                port_name_node1="purification_port_correct", port_name_node2="purification_port_correct2")
    return network

def setup_repeater_protocol(network, protocol_order, epsilon, purify="filter"):
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

    def start_on_success(protocol, start_subprotocol, success_subprotocol):
        # Convenience method to set subprotocol's start expression to be success of another
        protocol.subprotocols[start_subprotocol].start_expression = (
            protocol.subprotocols[start_subprotocol].await_signal(
                protocol.subprotocols[success_subprotocol], Signals.SUCCESS))
    
    def start_on_finished(protocol, start_subprotocol, success_subprotocol):
        # Convenience method to set subprotocol's start expression to be success of another
        protocol.subprotocols[start_subprotocol].start_expression = (
            protocol.subprotocols[start_subprotocol].await_signal(
                protocol.subprotocols[success_subprotocol], Signals.FINISHED))
                
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    end_nodes = [node for node in network.nodes.values() if END_NODE_BASENAME in node.name]
    switch_nodes = [node for node in network.nodes.values() if SWITCH_NODE_BASENAME in node.name]
    nodes=[end_nodes[0]] + switch_nodes + [end_nodes[1]]

    # Add CorrectProtocol to Bob
    subprotocol4 = CorrectProtocol(nodes[-1], len(nodes), purify=purify,protocol_order=protocol_order)
    protocol.add_subprotocol(subprotocol4)
    if protocol_order == 'PS':
        if len(nodes)==3:
            subprotocol1 = SwapProtocol(node=nodes[1], name=f"Swap_{nodes[1].name}", nodelist=nodes, purify=purify)
            protocol.add_subprotocol(subprotocol1)
            
            #For clarity, protocols are organized as node trios: (A)--(Ra,Rb)--(B) with R being repeater. Memory positions (and names) are numbered accordingly.
            #This means that whenever you encounter a pos, it's just numbering (0, 1, etc) but when you encounter 2*pos/2*pos+1 it's referencing a memory position.
            for pos in range(conf['mem_positions']//2 -1):
                purify = purify.lower()
                # Add entangle subprotocols
                pairs = 2 if purify == "distil" else 1
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[0], role="source", input_mem_pos=2*pos, num_pairs=pairs, name="entangle_A_{}".format(2*pos),qsource='qsource_0{}'.format(pos)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[1], role="source", input_mem_pos=2*pos, num_pairs=pairs, name="entangle_Rb_{}".format(2*pos),qsource='qsource_1{}'.format(pos)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[1], role="receiver", input_mem_pos=2*pos+1, num_pairs=pairs, name="entangle_Ra_{}".format(2*pos+1)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[2], role="receiver", input_mem_pos=2*pos+1, num_pairs=pairs, name="entangle_B_{}".format(2*pos+1)))
                # Setup all of the subprotocols
                if purify == "filter":
                    purify_cls, kwargs = Filter, {"epsilon": epsilon}
                else:
                    distil_role = "A"
                    purify_cls, kwargs = Distil, {"role": distil_role}
                #Add corresponding purification protocols
                for node1, port, name, distil_role in [
                    (nodes[0], 'purification_port_B_0_{}'.format(pos), "purify_A_{}".format(2*pos), "A"),
                    (nodes[1], 'purification_port_A_0_{}'.format(pos), "purify_Ra_{}".format(2*pos+1), "B"),
                    (nodes[2], 'purification_port_A_1_{}'.format(pos), "purify_B_{}".format(2*pos+1), "A"),
                    (nodes[1], 'purification_port_B_1_{}'.format(pos), "purify_Rb_{}".format(2*pos), "B")]:

                    protocol.add_subprotocol(purify_cls(
                            node1, port=node1.ports[port], name=name, **kwargs))

                # Set purify start expressions
                start_on_success(protocol, "purify_A_{}".format(2*pos),    "entangle_A_{}".format(2*pos))
                start_on_success(protocol, "purify_Ra_{}".format(2*pos+1),   "entangle_Ra_{}".format(2*pos+1))
                start_on_success(protocol, "purify_B_{}".format(2*pos+1),    "entangle_B_{}".format(2*pos+1))
                start_on_success(protocol, "purify_Rb_{}".format(2*pos),   "entangle_Rb_{}".format(2*pos))


                # Set entangle start expressions 
                start_expr_ent_RA = (protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].await_signal(
                    protocol.subprotocols["purify_Ra_{}".format(2*pos+1)], Signals.FAIL) |
                    protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].start_expression = start_expr_ent_RA
                start_expr_ent_RB = (protocol.subprotocols["entangle_Rb_{}".format(2*pos)].await_signal(
                    protocol.subprotocols["purify_Rb_{}".format(2*pos)], Signals.FAIL) |
                    protocol.subprotocols["entangle_Rb_{}".format(2*pos)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_Rb_{}".format(2*pos)].start_expression = start_expr_ent_RB
                start_expr_ent_A = (protocol.subprotocols["entangle_A_{}".format(2*pos)].await_signal(
                    protocol.subprotocols["purify_A_{}".format(2*pos)], Signals.FAIL) |
                    protocol.subprotocols["entangle_A_{}".format(2*pos)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_A_{}".format(2*pos)].start_expression = start_expr_ent_A
                start_expr_ent_B = (protocol.subprotocols["entangle_B_{}".format(2*pos+1)].await_signal(
                    protocol.subprotocols["purify_B_{}".format(2*pos+1)], Signals.FAIL) |
                    protocol.subprotocols["entangle_B_{}".format(2*pos+1)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_B_{}".format(2*pos+1)].start_expression = start_expr_ent_B

                
                
            events = [subprotocol1.await_signal(protocol.subprotocols[name], Signals.SUCCESS) for name in 
                    ["purify_Ra_{}".format(2*pos+1) for pos in range(conf['mem_positions']//2-1)] + 
                    ["purify_Rb_{}".format(2*pos) for pos in range(conf['mem_positions']//2-1)]]

            combined_events =  functools.reduce(lambda x, y: x & y, events)
            subprotocol1.start_expression = combined_events
            protocol.send_signal(Signals.WAITING)

        else:
            for i, node in zip(range(1, len(nodes), 2), nodes[1:-1:2]):
                nodelist=[nodes[i-1], nodes[i], nodes[i+1]]
                subprotocol1 = SwapProtocol(node=node, name=f"Swap_{node.name}", nodelist=nodelist)
                protocol.add_subprotocol(subprotocol1)

                #Protocolos de entrelazamiento
                for pos in range(conf['mem_positions']//2):
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[0], role="source", input_mem_pos=2*pos, num_pairs=1, name="entangle_A_{}_{}".format(i,pos+1),qsource='qsource_{}{}'.format(i-1,pos)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[1], role="source", input_mem_pos=2*pos, num_pairs=1, name="entangle_Rb_{}_{}".format(i,pos+1),qsource='qsource_{}{}'.format(i,pos)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[1], role="receiver", input_mem_pos=2*pos+1, num_pairs=1, name="entangle_Ra_{}_{}".format(i,pos+1)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[2], role="receiver", input_mem_pos=2*pos+1, num_pairs=1, name="entangle_B_{}_{}".format(i,pos+1)))

                    for node1, port, name in [ #, role 
                            (nodelist[0], 'purification_port_B_{}_{}'.format(i-1,pos), "purify_A_{}_{}".format(i,pos+1)),# 'A'),
                            #(nodes[0], 'purification_port_B_0_1', "purify_A_2"),# 'A'),
                            (nodelist[1], 'purification_port_A_{}_{}'.format(i-1,pos), "purify_Ra_{}_{}".format(i,pos+1)),# 'A'),
                            #(nodes[1], 'purification_port_A_0_1', "purify_Ra_2"),# 'A'),
                            (nodelist[2], 'purification_port_A_{}_{}'.format(i,pos), "purify_B_{}_{}".format(i,pos+1)),# 'B'),
                            #(nodes[2], 'purification_port_A_1_1', "purify_B_2"),# 'B'),
                            (nodelist[1], 'purification_port_B_{}_{}'.format(i,pos), "purify_Rb_{}_{}".format(i,pos+1))]:# 'B')
                            #(nodes[1], 'purification_port_B_1_1', "purify_Rb_2")]:#, 'B')]:

                        protocol.add_subprotocol(Filter(
                            node1, port=node1.ports[port], epsilon=epsilon, name=name))

                    #protocol.add_subprotocol(Distil(
                    #    node1, port=node1.ports[port], role=role, name=name))
                    # Set purify start expressions
                    start_on_success(protocol, "purify_A_{}_{}".format(i,pos+1),    "entangle_A_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_Ra_{}_{}".format(i,pos+1),   "entangle_Ra_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_B_{}_{}".format(i,pos+1),    "entangle_B_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_Rb_{}_{}".format(i,pos+1),   "entangle_Rb_{}_{}".format(i,pos+1))
                    #start_on_success(protocol, "purify_A_2",  "entangle_A_2")
                    #start_on_success(protocol, "purify_Ra_2", "entangle_Ra_2")
                    #start_on_success(protocol, "purify_B_2",  "entangle_B_2")
                    #start_on_success(protocol, "purify_Rb_2", "entangle_Rb_2")

                    # Set entangle start expressions 
                    start_expr_ent_RA = (protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_Ra_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_RA
                    start_expr_ent_RB = (protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_Rb_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_RB
                    start_expr_ent_A = (protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_A_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_A
                    start_expr_ent_B = (protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_B_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_B

                #guardar los resultados de estos SUCCESS para las posiciones que hay que medir en los repetidores
                subprotocol1.start_expression = (subprotocol1.await_signal(protocol.subprotocols['purify_Ra_{}_1'.format(i)], Signals.SUCCESS) & 
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Rb_{}_1'.format(i)], Signals.SUCCESS) &
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Ra_{}_2'.format(i)], Signals.SUCCESS) &
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Rb_{}_2'.format(i)], Signals.SUCCESS)
                                                    )
            for i, node in zip(range(2, len(nodes), 2), nodes[2:-1:2]):
                nodelist=[nodes[i-1], nodes[i], nodes[i+1]]
                subprotocol2 = SwapProtocol2(node=node, name=f"Swap2_{node.name}", nodelist=nodelist, protocol = protocol)
                protocol.add_subprotocol(subprotocol2)

            protocol.send_signal(Signals.WAITING)
    
    elif protocol_order == 'SP':
        
        if len(nodes)==3:
            subprotocol1 = SwapProtocol(node=nodes[1], name=f"Swap_{nodes[1].name}", nodelist=nodes, purify=purify)
            protocol.add_subprotocol(subprotocol1)
            
            #For clarity, protocols are organized as node trios: (A)--(Ra,Rb)--(B) with R being repeater. Memory positions (and names) are numbered accordingly.
            #This means that whenever you encounter a pos, it's just numbering (0, 1, etc) but when you encounter 2*pos/2*pos+1 it's referencing a memory position.
            for pos in range(conf['mem_positions']//2 -1):
                purify = purify.lower()
                # Add entangle subprotocols
                pairs = 2 if purify == "distil" else 1
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[0], role="source", input_mem_pos=2*pos, num_pairs=pairs, name="entangle_A_{}".format(2*pos),qsource='qsource_0{}'.format(pos)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[1], role="source", input_mem_pos=2*pos, num_pairs=pairs, name="entangle_Rb_{}".format(2*pos),qsource='qsource_1{}'.format(pos)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[1], role="receiver", input_mem_pos=2*pos+1, num_pairs=pairs, name="entangle_Ra_{}".format(2*pos+1)))
                protocol.add_subprotocol(EntangleNodes(
                    node=nodes[2], role="receiver", input_mem_pos=2*pos+1, num_pairs=pairs, name="entangle_B_{}".format(2*pos+1)))
                # Setup all of the subprotocols
                #if purify == "filter":
                #    purify_cls, kwargs = Filter, {"epsilon": epsilon}
                #else:
                    #distil_role = "A"
                    #purify_cls, kwargs = Distil, {"role": distil_role}
                #Add corresponding purification protocols
                for node1, port, name, distil_role in [
                        (nodes[0], 'purification_port_A_0_{}_SP'.format(pos), "purify_A_{}".format(2*pos), "A"),
                        (nodes[2], 'purification_port_B_0_{}_SP'.format(pos), "purify_B_{}".format(2*pos+1), "B")]:

                    protocol.add_subprotocol(Distil(
                            node1, port=node1.ports[port], name=name, role=distil_role))

                # Set purify start expressions
                start_on_success(protocol, "purify_A_{}".format(2*pos), subprotocol4.name)
                start_on_success(protocol, "purify_B_{}".format(2*pos+1), subprotocol4.name)

                # Set entangle start expressions 
                start_expr_ent_RA = (
                    (protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].await_signal(
                    protocol.subprotocols["purify_A_{}".format(2*pos)], Signals.FAIL) &
                    protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].await_signal(
                    protocol.subprotocols["purify_B_{}".format(2*pos+1)], Signals.FAIL) )|
                    protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_Ra_{}".format(2*pos+1)].start_expression = start_expr_ent_RA
                
                start_expr_ent_RB = (
                    (protocol.subprotocols["entangle_Rb_{}".format(2*pos)].await_signal(
                    protocol.subprotocols["purify_B_{}".format(2*pos+1)], Signals.FAIL) & 
                    protocol.subprotocols["entangle_Rb_{}".format(2*pos)].await_signal(
                    protocol.subprotocols["purify_A_{}".format(2*pos)], Signals.FAIL)) |
                    protocol.subprotocols["entangle_Rb_{}".format(2*pos)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_Rb_{}".format(2*pos)].start_expression = start_expr_ent_RB
                
                start_expr_ent_A = (protocol.subprotocols["entangle_A_{}".format(2*pos)].await_signal(
                    protocol.subprotocols["purify_A_{}".format(2*pos)], Signals.FAIL) |
                    protocol.subprotocols["entangle_A_{}".format(2*pos)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_A_{}".format(2*pos)].start_expression = start_expr_ent_A
                
                start_expr_ent_B = (protocol.subprotocols["entangle_B_{}".format(2*pos+1)].await_signal(
                    protocol.subprotocols["purify_B_{}".format(2*pos+1)], Signals.FAIL) |
                    protocol.subprotocols["entangle_B_{}".format(2*pos+1)].await_signal(protocol, Signals.WAITING))
                protocol.subprotocols["entangle_B_{}".format(2*pos+1)].start_expression = start_expr_ent_B

                
                
            events = [subprotocol1.await_signal(protocol.subprotocols[name], Signals.SUCCESS) for name in 
                    ["entangle_B_{}".format(2*pos+1) for pos in range(conf['mem_positions']//2-1)] + 
                    ["entangle_A_{}".format(2*pos) for pos in range(conf['mem_positions']//2-1)] + 
                    ["entangle_Rb_{}".format(2*pos) for pos in range(conf['mem_positions']//2-1)] +
                    ["entangle_Ra_{}".format(2*pos+1) for pos in range(conf['mem_positions']//2-1)] ]

            combined_events =  functools.reduce(lambda x, y: x & y, events)
            subprotocol1.start_expression = combined_events
            protocol.send_signal(Signals.WAITING)

        else:
            for i, node in zip(range(1, len(nodes), 2), nodes[1:-1:2]):
                nodelist=[nodes[i-1], nodes[i], nodes[i+1]]
                subprotocol1 = SwapProtocol(node=node, name=f"Swap_{node.name}", nodelist=nodelist)
                protocol.add_subprotocol(subprotocol1)

                #Protocolos de entrelazamiento
                for pos in range(conf['mem_positions']//2):
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[0], role="source", input_mem_pos=2*pos, num_pairs=1, name="entangle_A_{}_{}".format(i,pos+1),qsource='qsource_{}{}'.format(i-1,pos)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[1], role="source", input_mem_pos=2*pos, num_pairs=1, name="entangle_Rb_{}_{}".format(i,pos+1),qsource='qsource_{}{}'.format(i,pos)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[1], role="receiver", input_mem_pos=2*pos+1, num_pairs=1, name="entangle_Ra_{}_{}".format(i,pos+1)))
                    protocol.add_subprotocol(EntangleNodes(
                        node=nodelist[2], role="receiver", input_mem_pos=2*pos+1, num_pairs=1, name="entangle_B_{}_{}".format(i,pos+1)))

                    for node1, port, name in [ #, role 
                            (nodelist[0], 'purification_port_B_{}_{}'.format(i-1,pos), "purify_A_{}_{}".format(i,pos+1)),# 'A'),
                            #(nodes[0], 'purification_port_B_0_1', "purify_A_2"),# 'A'),
                            (nodelist[1], 'purification_port_A_{}_{}'.format(i-1,pos), "purify_Ra_{}_{}".format(i,pos+1)),# 'A'),
                            #(nodes[1], 'purification_port_A_0_1', "purify_Ra_2"),# 'A'),
                            (nodelist[2], 'purification_port_A_{}_{}'.format(i,pos), "purify_B_{}_{}".format(i,pos+1)),# 'B'),
                            #(nodes[2], 'purification_port_A_1_1', "purify_B_2"),# 'B'),
                            (nodelist[1], 'purification_port_B_{}_{}'.format(i,pos), "purify_Rb_{}_{}".format(i,pos+1))]:# 'B')
                            #(nodes[1], 'purification_port_B_1_1', "purify_Rb_2")]:#, 'B')]:

                        protocol.add_subprotocol(Filter(
                            node1, port=node1.ports[port], epsilon=epsilon, name=name))

                    #protocol.add_subprotocol(Distil(
                    #    node1, port=node1.ports[port], role=role, name=name))
                    # Set purify start expressions
                    start_on_success(protocol, "purify_A_{}_{}".format(i,pos+1),    "entangle_A_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_Ra_{}_{}".format(i,pos+1),   "entangle_Ra_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_B_{}_{}".format(i,pos+1),    "entangle_B_{}_{}".format(i,pos+1))
                    start_on_success(protocol, "purify_Rb_{}_{}".format(i,pos+1),   "entangle_Rb_{}_{}".format(i,pos+1))
                    #start_on_success(protocol, "purify_A_2",  "entangle_A_2")
                    #start_on_success(protocol, "purify_Ra_2", "entangle_Ra_2")
                    #start_on_success(protocol, "purify_B_2",  "entangle_B_2")
                    #start_on_success(protocol, "purify_Rb_2", "entangle_Rb_2")

                    # Set entangle start expressions 
                    start_expr_ent_RA = (protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_Ra_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_Ra_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_RA
                    start_expr_ent_RB = (protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_Rb_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_Rb_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_RB
                    start_expr_ent_A = (protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_A_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_A_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_A
                    start_expr_ent_B = (protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].await_signal(
                        protocol.subprotocols["purify_B_{}_{}".format(i,pos+1)], Signals.FAIL) |
                        protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].await_signal(protocol, Signals.WAITING))
                    protocol.subprotocols["entangle_B_{}_{}".format(i,pos+1)].start_expression = start_expr_ent_B

                #guardar los resultados de estos SUCCESS para las posiciones que hay que medir en los repetidores
                subprotocol1.start_expression = (subprotocol1.await_signal(protocol.subprotocols['purify_Ra_{}_1'.format(i)], Signals.SUCCESS) & 
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Rb_{}_1'.format(i)], Signals.SUCCESS) &
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Ra_{}_2'.format(i)], Signals.SUCCESS) &
                                                    subprotocol1.await_signal(protocol.subprotocols['purify_Rb_{}_2'.format(i)], Signals.SUCCESS)
                                                    )
            for i, node in zip(range(2, len(nodes), 2), nodes[2:-1:2]):
                nodelist=[nodes[i-1], nodes[i], nodes[i+1]]
                subprotocol2 = SwapProtocol2(node=node, name=f"Swap2_{node.name}", nodelist=nodelist, protocol = protocol)
                protocol.add_subprotocol(subprotocol2)

            protocol.send_signal(Signals.WAITING)
    
    return protocol        

def setup_datacollector(network, protocol, protocol_order):
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
        qubit_a1, = nodes[0].qmemory.peek([0])
        qubit_a2, = nodes[0].qmemory.peek([2])
        position_b=protocol.subprotocols["purify_B_1"].get_signal_result(Signals.SUCCESS) if 'Distil' in str([k for k in ConstrainedMapView(protocol.subprotocols,[ValueConstraint(lambda x: 'purify' in x.name)]).values()]) else final_secpos
        qubit_b, = nodes[-1].qmemory.peek([position_b])
        if qubit_a1 is None:
            fidelity =ns.qubits.fidelity([qubit_b, qubit_a2], ks.b00, squared=True)
        else:
            fidelity =ns.qubits.fidelity([qubit_b, qubit_a1], ks.b00, squared=True)
            
        fidelities.append(fidelity)
        print(fidelity)
        ns.sim_stop()
        protocol.send_signal(Signals.WAITING)
        return {"fidelity": fidelity}#,"fidelity 2": fidelity2,"Highest fidelity ch1":fidelityc1, "Highest fidelity ch2":fidelityc2}

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    if protocol_order == 'PS':
        dc.collect_on([pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value)])
    elif protocol_order == 'SP':
        dc.collect_on([pydynaa.EventExpression(source=protocol.subprotocols["purify_B_1"],
                                          event_type=Signals.SUCCESS.value)])
    
    return dc

def run_simulation(epsilon, num_nodes, node_distance, num_iters, purify, protocol_order):
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
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    
    network = setup_network(num_nodes, node_distance=node_distance,
                            source_frequency=conf['source_frequency'] / est_runtime)

    protocol = setup_repeater_protocol(network, protocol_order, epsilon=epsilon, purify=purify)
    dc = setup_datacollector(network, protocol, protocol_order)
    protocol.start()
    #ns.sim_run()
    ns.sim_run(10*est_runtime*num_iters)
    #ic(dc.dataframe)


    #print(f" Average fidelity of {dc.dataframe['fidelity'].mean():.3f} with a STD of {dc.dataframe['fidelity'].std():.3f}")
    return dc.dataframe


#Read properties
conf = {}
readProperties(conf,'properties.cfg')

ic(conf)

ns.set_qstate_formalism(ns.QFormalism.DM)

def create_plot():
    """Run the simulation for multiple nodes and distances and show them in a figure.

    Parameters
    ----------

    num_iters : int, optional
        Number of iterations per simulation configuration.
        At least 1. Default 2000.
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    values = ['distil','filter']

    data_diff = pandas.DataFrame()
    for eps in values:
        data = pandas.DataFrame()
        for i in range(conf['num_iters']):
            data[i] = run_simulation(epsilon=1,
                                    num_nodes=conf['num_nodes'],
                                    node_distance=conf['node_distance'],
                                    num_iters=1,
                                    purify=eps)['fidelity']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        data.plot(y='fidelity', yerr='sem', label=f"{eps}", ax=ax)
        data_diff[eps] = data['fidelity']
    # Calculate the difference between the variables
    data_diff['diff'] = abs(data_diff['filter'] - data_diff['distil'])
    # Plot difference
    #data_diff.plot(y='diff', label='Difference', color='black', linestyle='--', ax=ax)

    plt.xlabel("Iteration")
    current_ticks = plt.gca().get_xticks()
    new_ticks = np.arange(0, max(current_ticks) , step=10)
    plt.xticks(new_ticks)
    plt.ylabel("Fidelity")
    plt.ylim(0, 1)
    plt.title("Repeater chain with different total lengths")
    plt.show()

def create_plot1():
    """Run the simulation for multiple nodes and distances and show them in a figure.

    Parameters
    ----------

    num_iters : int, optional
        Number of iterations per simulation configuration.
        At least 1. Default 2000.
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from cycler import cycler
    
    fig, ax = plt.subplots()
        # Retrieve the default color cycle from Pandas
    pandas_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Set the color cycle in Matplotlib
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=pandas_color_cycle)
    values = ['distil','filter']
    titles=['Distillation', 'Routing']
    protocol = ['SP','PS']
    data_diff = pandas.DataFrame()
    data_sem = pandas.DataFrame()
    for eps,title in zip(values,titles):
        data = pandas.DataFrame()
        for i in range(conf['num_iters']):
            data[i] = run_simulation(epsilon=1,
                                    num_nodes=conf['num_nodes'],
                                    node_distance=conf['node_distance'],
                                    num_iters=100,
                                    purify=eps,
                                    protocol_order=protocol[0])['fidelity']
            # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        #data.plot(y='fidelity',yerr='sem', label=f"{title}", ax=ax)
        data_diff[eps] = data['fidelity']
        data_sem[eps] = data['sem']
        #plt.errorbar(data.index.tolist(), data_diff[eps], data_sem[eps], marker= 's', linestyle= 'None')
    # Calculate the difference between the variables
    #data_diff['diff'] = abs(data_diff['filter'] - data_diff['distil'])
    mean_fid_distill = data_diff['distil'].mean()
    std_fid_distill = data_diff['distil'].std()
    mean_fid_raw = data_diff['filter'].mean()
    std_fid_raw = data_diff['filter'].std()
    # Plot difference
    #data_diff.plot(y='diff', label='Difference', color='black', linestyle='--', ax=ax)

    ax.axhline(mean_fid_distill, color='blue', label=f'Mean fidelity (distil): {mean_fid_distill:.3f}')
    ax.fill_between(range(0, 50), mean_fid_distill - std_fid_distill, mean_fid_distill + std_fid_distill,
                    color='blue', alpha=0.2)    
    ax.axhline(mean_fid_raw, color='orange', label=f'Mean fidelity (raw): {mean_fid_raw:.3f}')
    ax.fill_between(range(0, 50), mean_fid_raw - std_fid_raw, mean_fid_raw + std_fid_raw,
                    color='orange', alpha=0.2)     
    ax.errorbar(data.index.tolist(), data_diff['distil'], yerr= data_sem['distil'], marker='o', linestyle='None', label='Distilled')
    ax.errorbar(data.index.tolist(), data_diff['filter'], yerr= data_sem['filter'], marker='s', linestyle='None', label= 'Raw')

    plt.xlabel("Iteration")
    current_ticks = plt.gca().get_xticks()
    new_ticks = np.arange(0, max(current_ticks) , step=10)
    plt.xticks(new_ticks)
    plt.ylabel("Fidelity")
    plt.legend()
    plt.ylim(0.5, 1)
    plt.xlim(0,50)
    plt.title("Repeater chain with distillation improvement")
    plt.savefig('Distil_bis.pdf', format='pdf', dpi=300)
    plt.show()
    
create_plot1()
