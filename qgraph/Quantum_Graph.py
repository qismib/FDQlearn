import pennylane as qml
from pennylane import numpy as np
import torch


"""
NODE ENCODING LAYER
"""


def RX_layer(the_G):
    """
    :param: the_G: graph representing the Feynamn diagram
    :return: None
    """

    """
    ANGLES I USE TO ENCODE THE NODE FEATURES (IF A NODE IS A FINAL, INITIAL OR INTERMEDIATE NODE OF THE FEYNMAN DIAGRAM)
    """
    encoding_angles = np.array([np.pi/2, np.pi, 3*np.pi/2])

    for the_node in the_G.nodes:
        qml.RX(np.inner(encoding_angles, the_G.nodes[the_node]['state']),
               wires=the_node)


"""
LAYER THAT ENCODE THE EDGE FEATURES, I USE A ZZ GATE TO CREATE ENTANGLEMENT INTO THE CIRCUIT ANDO MOREOVER TO 
ENCODE IN A SORT OF WAY THE TOPOLOGY OF THE INPUT DIAGRAM (CLASSICALLY ENCODED AS A GRAPH)
"""


def ZZ_layer(the_G, the_feature: str):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_feature: string of the edge feature I want to encode in this layer
    :return: None
    """
    for the_edge in the_G.edges:
        the_feat = the_G.edges[the_edge][the_feature]
        qml.IsingZZ(the_feat, wires=the_edge)


"""
PARAMETRIC VERSION OF ZZ_LAYER, HERE I MULTIPLY EACH EDGE FEATURE BY A TRAINABLE PARAMETER
"""


def parametric_ZZ_layer(the_G, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: trainable parameters that weights the edge features
    :return: None
    """
    for the_edge in the_G.edges:
        the_feat = torch.tensor([the_G.edges[the_edge]['mass'], the_G.edges[the_edge]['spin'],
                                 the_G.edges[the_edge]['charge']], dtype=torch.float)
        qml.IsingZZ(torch.inner(the_feat, the_params), wires=the_edge)


def fully_parametric_ZZ_layer(the_G, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: trainable parameters that weights the edge features
    :return: None
    """
    photon_params = the_params[:3]
    electron_params = the_params[3:6]
    muon_params = the_params[6:]

    for the_edge in the_G.edges:
        the_feat = torch.tensor([the_G.edges[the_edge]['mass'], the_G.edges[the_edge]['spin'],
                                 the_G.edges[the_edge]['charge']], dtype=torch.float)
        if the_feat[0] == 0:
            qml.IsingZZ(torch.inner(the_feat, photon_params), wires=the_edge)
        elif 0.5 < the_feat[0] < 0.6:
            qml.IsingZZ(torch.inner(the_feat, electron_params), wires=the_edge)
        elif 105 < the_feat[0] < 106:
            qml.IsingZZ(torch.inner(the_feat, muon_params), wires=the_edge)


"""
ROTATION DEPENDENT ON 2 ANGLES, MOMENTUM AND SCATTERING ANGLES, THEY ENCODE THE KINEMATIC PART OF THE CALCULATION
"""


def Kinetic_layer(the_G):
    """
    :param: the_G: graph representing the Feynamn diagram
    :return: None
    """
    for the_node in the_G.nodes:
        # qml.U2(the_G.graph['p_norm'], the_G.graph['theta'], wires=the_node)  # for a circuit with momentum
        qml.U1(the_G.graph['theta'], wires=the_node) # for a circuit without momentum


########################################################################################################################


"""
FEATURE MAP FOR ENCODING THE I-TH GRAPH G OF THE DATASET; 
COMPOSED BY ALTERNATING ZZ_LAYER AND RX_LAYER FOR THREE TIMES (NUMBER OF EDGE FEATURES),
AND BY INSERTING THE KINETIC LAYER AT THE END
"""


def qgnn_feature_map(the_G):
    """
    :param: the_G: graph representing the Feynamn diagram
    :return: None
    """

    for i in ['mass', 'spin', 'charge']:

        for j in the_G.nodes:
            qml.Hadamard(wires=j)

        qml.Barrier()

        # encode edge's features
        ZZ_layer(the_G, i)

        qml.Barrier()

        # encode node's features
        RX_layer(the_G)

        qml.Barrier()
    # encode angle and momentum features
    Kinetic_layer(the_G)

    qml.Barrier()


"""
FEATURE MAP FOR ENCODING THE I-TH GRAPH G OF THE DATASET; 
COMPOSED BY ALTERNATING ZZ_LAYER AND RX_LAYER ONCE, BY MULTIPLYING EACH EDGE FEATURE BY A
TRAINABLE PARAMETER, AND BY INSERTING THE KINETIC LAYER AT THE END
"""


def parametric_qgnn_feature_map(the_G, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: number of trainable parameters that weights the edge features
    :return: None
    """
    assert len(the_params) == len(['mass', 'spin', 'charge']), 'parameters must be equal to the number of edge features'

    for j in the_G.nodes:
        qml.Hadamard(wires=j)

    qml.Barrier()

    # encode edge's features
    parametric_ZZ_layer(the_G, the_params=the_params)

    qml.Barrier()

    # encode node's features
    RX_layer(the_G)

    qml.Barrier()

    # encode angle and momentum features
    Kinetic_layer(the_G)

    qml.Barrier()


def fully_parametric_qgnn_feature_map(the_G, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: number of trainable parameters that weights the edge features
    :return: None
    """
    assert len(the_params) == 3*len(['mass', 'spin', 'charge']), 'parameters must be equal to the number of edge ' \
                                                                 'times the number of particles (3) features'

    for j in the_G.nodes:
        qml.Hadamard(wires=j)

    qml.Barrier()

    # encode edge's features
    fully_parametric_ZZ_layer(the_G, the_params=the_params)

    qml.Barrier()

    # encode node's features
    RX_layer(the_G)

    qml.Barrier()

    # encode angle and momentum features
    Kinetic_layer(the_G)

    qml.Barrier()

########################################################################################################################


"""
ANSATZ CIRCUIT FOR THE QGNN
"""


def qgnn_ansatz(the_G, the_n_layers, the_params):
    """
    Trainable ansatz having l * (m + n + 2) parameters, where m number of arcs, n is the number of vertices,
    # l is the number of layers and +2 for theta and p parameters
    :param: the_G: graph representing a feynman diagram
    :param: the_n_layers: number of layers (depth of the circuit)
    :param: the_params: value of the parameters
    """
    the_m_init = 0
    the_m_fin = 0
    the_m_prop = 0
    the_in_states = []
    the_fin_states = []
    the_prop = []

    for node in the_G.nodes:
        if the_G.nodes[node]['state'][0] == 1:  # if the state is the initial state
            the_m_init += 1
            the_in_states.append(node)
        elif the_G.nodes[node]['state'][2] == 1:  # if the state is the final state
            the_m_fin += 1
            the_fin_states.append(node)
        else:  # if the state is the propagator state
            the_m_prop += 1
            the_prop.append(node)

    the_n = len(the_G.nodes)
    the_m = int((the_m_init + the_m_fin + (the_m_prop - 1) / 2) * the_m_prop)
    # I connect each initial node to any possible propagator,
    # the same for the final nodes

    # number of parameters with the momentum p
    # assert len(the_params) == the_n_layers * (the_m + the_n + 2), "Number of parameters is wrong"

    # number of parameters without the momentum p
    assert len(the_params) == the_n_layers * (the_m + the_n + 1), "Number of parameters is wrong"

    for i in range(the_n_layers):
        # here I divide the_layer_params list into a list for parameters that will act on edges,
        # parameters that will act on nodes and the ones for momentum and angle features
        the_layer_params = the_params[i * (the_m + the_n + 1):(i + 1) * (the_m + the_n + 1)]  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1
        the_edge_params = the_layer_params[:the_m]
        the_nodes_params = the_layer_params[the_m:-1]  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1
        the_kinetic_params = the_layer_params[-1:]  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1

        # ind is the index of the_edge_params, trainable parameters.
        ind = 0

        # I'm building parametrized ZZ-gates that connect each initial state (node feature=[1,0,0])
        # to every propagator state (node feature = [0,1,0])
        for x in the_in_states:
            for y in the_prop:
                qml.IsingZZ(the_edge_params[ind], wires=(x, y))
                ind += 1
        """
        In Feynman Diagrams with only 1 loop I only have 2 propagation-nodes and
        only one way I can connect them; for multi-loops Diagrams I need all the combinations
        I can connect the various propagator nodes 
        """

        # I'm building parametrized ZZ-gate that connects the two propagator states (node feature=[0,1,0])
        qml.IsingZZ(the_edge_params[ind], wires=(the_prop[0], the_prop[1]))
        ind += 1

        # I'm building parametrized ZZ-gates that connect each final state (node feature=[0,0,1])
        # to every propagator state (node feature = [0,1,0])
        for x in the_fin_states:
            for y in the_prop:
                qml.IsingZZ(the_edge_params[ind], wires=(y, x))
                ind += 1

        # for each node of the graph I perform a parametrized X-rotation and a parametrized
        # U2-gate (rotation and phase-shift on a single qubit)
        for j in the_G.nodes:
            qml.RX(the_nodes_params[j], wires=j)
            # qml.U2(the_kinetic_params[0], the_kinetic_params[1], wires=j)  # for a circuit with momentum
            qml.U1(the_kinetic_params[0], wires=j)  # for a circuit without momentum


########################################################################################################################


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH UNPARAMETRIZED FEATURE MAP)
"""


def qgnn(the_G, the_n_layers, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # the feature map
    qgnn_feature_map(the_G)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, the_params)


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH PARAMETRIZED FEATURE MAP)
"""


def parametric_qgnn(the_G, the_n_layers, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # defining parameters for feature map(first three) and ansatz (the rest)
    feat_params = the_params[:3]
    ansatz_params = the_params[3:]

    # the parametric feature map
    parametric_qgnn_feature_map(the_G, feat_params)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, ansatz_params)


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH FULLY PARAMETRIZED FEATURE MAP)
"""


def fully_parametric_qgnn(the_G, the_n_layers, the_params):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # defining parameters for feature map(first three) and ansatz (the rest)
    feat_params = the_params[:9]
    ansatz_params = the_params[9:]

    # the parametric feature map
    fully_parametric_qgnn_feature_map(the_G, feat_params)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, ansatz_params)


########################################################################################################################

def bhabha_operator(the_wire=0, a=torch.tensor(2., dtype=torch.float), b=torch.tensor(1., dtype=torch.float)):
    """
        :param: the_wire: qubit on which the operator acts
        :param: a, b: positive real coefficient
        :return: an operator which is diagonal, hermitian and positive definite
        """
    assert a != b, "a and b must be different"
    return torch.abs(a)*qml.Projector(basis_state=[0], wires=the_wire) + torch.abs(b)*qml.Projector(basis_state=[1], wires=the_wire)

########################################################################################################################


dev1 = qml.device("default.qubit", wires=6)


@qml.qnode(dev1, interface='torch', diff_method="parameter-shift")
def expect_value(the_G, the_n_layers, the_params, the_choice):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: the_choice: string that tells which feature map we want to use
    :return: expectation value of a diagonal, positive hermitian operator on the first qubit
    """
    the_circuit_params = the_params[:-2]
    the_observable_params = the_params[-2:]

    if the_choice == 'parametrized':
        parametric_qgnn(the_G, the_n_layers, the_circuit_params)
    elif the_choice == 'unparametrized':
        qgnn(the_G, the_n_layers, the_circuit_params)
    elif the_choice == 'fully_parametrized':
        fully_parametric_qgnn(the_G, the_n_layers, the_circuit_params)
    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    my_operator = bhabha_operator(0, the_observable_params[0], the_observable_params[1]) #this operator is the one we want to define for the interference circuit
    return qml.expval(my_operator)


########################################################################################################################


# I'm working ong tree level QED with simple Feynman diagram, the number of vertices of the diagram is fixed and equal
# to 6, then I add one more qubit as a control qubit

"""
Circuit that extract the squared total matrix element
of Bhabha scattering (generalizable to other scattering processes)
"""
dev2 = qml.device("default.qubit", wires=7)


@qml.qnode(dev2, interface='torch', diff_method="parameter-shift")
def total_matrix_circuit(the_s_channel, the_s_params, the_t_channel, the_t_params, the_layers,
                         the_choice: str = 'parametrized'):
    """
    Circuit that extract the squared total matrix element of 2 Feynman Diagrams
    :param: the_s_channel: graph representing the s-channel diagram
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_t_channel: graph representing the t-channel diagram
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :return: expectation value of a composite operator that is the prediction of the matrix element squared
    """

    the_s_observable = the_s_params[-2:]
    the_s_params = the_s_params[:-2]
    the_t_observable = the_t_params[-2:]
    the_t_params = the_t_params[:-2]

    qml.Hadamard(wires=6)

    if the_choice == 'parametrized':

        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    elif the_choice == 'unparametrized':
        qml.ctrl(qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    elif the_choice == 'fully_parametrized':
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    qml.Hadamard(wires=6)

    alpha = the_s_observable[0]*the_t_observable[0]
    beta = the_s_observable[1]*the_s_observable[1]
    my_operator = bhabha_operator(0, torch.sqrt(alpha), torch.sqrt(beta))

    return qml.expval(my_operator @ qml.Projector(basis_state=[0], wires=6))


"""
Circuit that extract the interference term of Bhabha scattering (generalizable to other 
scattering processes)
"""
dev3 = qml.device("default.qubit", wires=7)


@qml.qnode(dev3, interface='torch', diff_method="parameter-shift")
def interference_circuit(the_s_channel, the_s_params, the_t_channel, the_t_params, the_layers,
                         the_choice: str = 'parametrized'):
    """
    Circuit that extract the interference term of 2 Feynman Diagrams
    :param: the_s_channel: graph representing the s-channel diagram
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_t_channel: graph representing the t-channel diagram
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :return: expectation value of a composite operator that is the prediction of the interference
    """

    the_s_observable = the_s_params[-2:]
    the_s_params = the_s_params[:-2]
    the_t_observable = the_t_params[-2:]
    the_t_params = the_t_params[:-2]

    qml.Hadamard(wires=6)

    if the_choice == 'parametrized':

        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    elif the_choice == 'unparametrized':
        qml.ctrl(qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    elif the_choice == 'fully_parametrized':
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params)
        qml.PauliX(wires=6)
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params)

    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    qml.Hadamard(wires=6)

    alpha = the_s_observable[0] * the_t_observable[0]
    beta = the_s_observable[1] * the_s_observable[1]
    my_operator = bhabha_operator(0, torch.sqrt(alpha), torch.sqrt(beta))

    return qml.expval(my_operator @ qml.PauliZ(wires=6))
