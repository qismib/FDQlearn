import pennylane as qml
from pennylane import numpy as np
import torch
from pennylane.ops.op_math import Sum


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
NODE ENCODING LAYER WITH CONTROL OPERATORS, WHICH GIVE MORE IDEA OF THE TIME FLOW OVER THE DIAGRAM
"""


def Node_encoding(the_G):
    """
       :param: the_G: graph representing the Feynamn diagram
       :return: None
    """

    init_node = []
    prop_node = []
    final_node = []

    encoding_angles = np.array([np.pi/2, np.pi])

    for node in the_G.nodes:
        if the_G.nodes[node]['state'][0] == 1:  # if the state is the initial state
            init_node.append(node)
        elif the_G.nodes[node]['state'][2] == 1:  # if the state is the final state
            final_node.append(node)
        else:  # if the state is the propagator state
            prop_node.append(node)

    for i in prop_node:
        qml.ctrl(qml.RX, control=init_node)(encoding_angles[0], wires=i)

    for j in final_node:
        qml.ctrl(qml.RX, control=prop_node)(encoding_angles[1], wires=j)


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


"""
PARAMETRIC ZZ LAYER WHERE I USE DIFFERENT SET OF TRAINABLE PARAMETERS FOR EACH TYPE OF PARTICLE OF THE INTERACTION
"""


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


def Kinetic_layer(the_G, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    if massive is True:
        for the_node in the_G.nodes:
            qml.U2(the_G.graph['p_norm'], the_G.graph['theta'], wires=the_node)  # for a circuit with momentum

    else:
        for the_node in the_G.nodes:
            qml.U1(the_G.graph['theta'], wires=the_node) # for a circuit without momentum


########################################################################################################################


"""
FEATURE MAP FOR ENCODING THE I-TH GRAPH G OF THE DATASET; 
COMPOSED BY ALTERNATING ZZ_LAYER AND RX_LAYER FOR THREE TIMES (NUMBER OF EDGE FEATURES),
AND BY INSERTING THE KINETIC LAYER AT THE END
"""


def qgnn_feature_map(the_G, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: massive: boolean value that indicates whether we're in massive or massless regime
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
        # Node_encoding(the_G)

        qml.Barrier()
    # encode angle and momentum features
    Kinetic_layer(the_G, massive)

    qml.Barrier()


"""
FEATURE MAP FOR ENCODING THE I-TH GRAPH G OF THE DATASET; 
COMPOSED BY ALTERNATING ZZ_LAYER AND RX_LAYER ONCE, BY MULTIPLYING EACH EDGE FEATURE BY A
TRAINABLE PARAMETER, AND BY INSERTING THE KINETIC LAYER AT THE END
"""


def parametric_qgnn_feature_map(the_G, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: number of trainable parameters that weights the edge features
    :param: massive: boolean value that indicates whether we're in massive or massless regime
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
    # Node_encoding(the_G)

    qml.Barrier()

    # encode angle and momentum features
    Kinetic_layer(the_G, massive)

    qml.Barrier()


"""
PARAMETRIC FEATURE MAP WHERE I USE THE FULLY_PARAMETRIC_ZZ_LAYER FUNCTION
"""


def fully_parametric_qgnn_feature_map(the_G, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_params: number of trainable parameters that weights the edge features
    :param: massive: boolean value that indicates whether we're in massive or massless regime
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
    # Node_encoding(the_G)

    qml.Barrier()

    # encode angle and momentum features
    Kinetic_layer(the_G, massive)

    qml.Barrier()

########################################################################################################################


"""
ANSATZ CIRCUIT FOR THE QGNN
"""


def qgnn_ansatz(the_G, the_n_layers, the_params, massive: bool = False):
    """
    Trainable ansatz having l * (m + n + 2) parameters, where m number of arcs, n is the number of vertices,
    # l is the number of layers and +2 for theta and p parameters
    :param: the_G: graph representing a feynman diagram
    :param: the_n_layers: number of layers (depth of the circuit)
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
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

    # I put kinetic_params = 1 if we're in the massless regime
    # I put kinetic_params = 2 if we're in the massive regime
    if massive is False:
        kinetic_num = 1
    else:
        kinetic_num = 2

    the_n = len(the_G.nodes)
    the_m = int((the_m_init + the_m_fin + (the_m_prop - 1) / 2) * the_m_prop)
    # I connect each initial node to any possible propagator,
    # the same for the final nodes

    assert len(the_params) == the_n_layers * (the_m + the_n + kinetic_num), "Number of parameters is wrong"

    for i in range(the_n_layers):
        # here I divide the_layer_params list into a list for parameters that will act on edges,
        # parameters that will act on nodes and the ones for momentum and angle features
        the_layer_params = the_params[i * (the_m + the_n + kinetic_num):(i + 1) * (the_m + the_n + kinetic_num)]  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        the_edge_params = the_layer_params[:the_m]
        the_nodes_params = the_layer_params[the_m:-kinetic_num]  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        the_kinetic_params = the_layer_params[-kinetic_num:]  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

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

            if kinetic_num == 2:
                qml.U2(the_kinetic_params[0], the_kinetic_params[1], wires=j)  # for a circuit with momentum

            elif kinetic_num == 1:
                qml.U1(the_kinetic_params[0], wires=j)  # for a circuit without momentum


"""
FULLY CONNECTED ANSATZ CIRCUIT FOR THE QGNN
"""


def fully_connected_ansatz(the_G, the_n_layers, the_params, massive: bool = False):
    """
    Trainable fully connected ansatz having l * (n*(n-1)/2 +n + 1 (2)) parameters, where n is the number of vertices,
    # l is the number of layers and +1 (+2) for theta (and p) parameters
    :param: the_G: graph representing a feynman diagram
    :param: the_n_layers: number of layers (depth of the circuit)
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    # I put kinetic_params = 1 if we're in the massless regime
    # I put kinetic_params = 2 if we're in the massive regime
    if massive is False:
        kinetic_num = 1
    else:
        kinetic_num = 2

    # number of edges of the graph
    the_n = len(the_G.nodes)
    permutations = the_n * (the_n - 1) // 2

    assert len(the_params) == the_n_layers * (permutations + the_n + kinetic_num), "Number of parameters is wrong"

    for i in range(the_n_layers):
        # here I divide the_layer_params list into a list for parameters that will act on edges,
        # parameters that will act on nodes and the ones for momentum and angle features
        the_layer_params = the_params[i * (permutations + the_n + kinetic_num):(i + 1) * (permutations + the_n + kinetic_num)]  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        the_edge_params = the_layer_params[:permutations]
        the_nodes_params = the_layer_params[permutations:-kinetic_num]  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        the_kinetic_params = the_layer_params[-kinetic_num:]  #  KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

        # ind is the index of the_edge_params, trainable parameters.
        ind = 0

        # I'm building parametrized ZZ-gates that connect each initial state (node feature=[1,0,0])
        # to every propagator state (node feature = [0,1,0])
        for node in range(the_n-1):
            for comb in range(node+1, the_n):
                qml.IsingZZ(the_edge_params[ind], wires=(node, comb))
                ind += 1

        # for each node of the graph I perform a parametrized X-rotation and a parametrized
        # U2-gate (rotation and phase-shift on a single qubit)
        for j in the_G.nodes:
            qml.RX(the_nodes_params[j], wires=j)

            if kinetic_num == 2:
                qml.U2(the_kinetic_params[0], the_kinetic_params[1], wires=j)  # for a circuit with momentum

            elif kinetic_num == 1:
                qml.U1(the_kinetic_params[0], wires=j)  # for a circuit without momentum


########################################################################################################################


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH UNPARAMETRIZED FEATURE MAP)
"""


def qgnn(the_G, the_n_layers, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # the feature map
    qgnn_feature_map(the_G, massive)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, the_params, massive)


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH PARAMETRIZED FEATURE MAP)
"""


def parametric_qgnn(the_G, the_n_layers, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # defining parameters for feature map(first three) and ansatz (the rest)
    feat_params = the_params[:3]
    ansatz_params = the_params[3:]

    # the parametric feature map
    parametric_qgnn_feature_map(the_G, feat_params, massive)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, ansatz_params, massive)


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH FULLY PARAMETRIZED FEATURE MAP)
"""


def fully_parametric_qgnn(the_G, the_n_layers, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # defining parameters for feature map(first three) and ansatz (the rest)
    feat_params = the_params[:9]
    ansatz_params = the_params[9:]

    # the parametric feature map
    fully_parametric_qgnn_feature_map(the_G, feat_params, massive)

    # the ansatz
    qgnn_ansatz(the_G, the_n_layers, ansatz_params, massive)


"""
FUNCTION THAT DEFINE THE GLOBAL QUANTUM CIRCUIT (WITH FULLY CONNECTED ANSATZ)
"""


def fully_connected_qgnn(the_G, the_n_layers, the_params, massive: bool = False):
    """
    :param: the_G: graph representing the Feynamn diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    # get the number of vertices
    the_n_wires = len(the_G.nodes)

    # defining parameters for feature map(first three) and ansatz (the rest)
    feat_params = the_params[:3]
    ansatz_params = the_params[3:]

    # the parametric feature map
    parametric_qgnn_feature_map(the_G, feat_params, massive)

    # the ansatz
    fully_connected_ansatz(the_G, the_n_layers, ansatz_params, massive)


########################################################################################################################


"""
FUNCTION THAT DEFINE THE OBSERVABLE WE USE TO MAKE THE MEASURE FOR THE INTERFERENCE CIRCUIT
"""


def bhabha_operator(the_wire=0, a=torch.tensor(2., dtype=torch.float, requires_grad=False),
                    b=torch.tensor(1., dtype=torch.float, requires_grad=False)):
    """
    :param: the_wire: qubit on which the operator acts
    :param: a, b: positive real coefficient
    :return: an operator which is diagonal, hermitian and positive definite
    """

    assert a != b, "a and b must be different"
    a = np.array(a.detach().numpy(), requires_grad=False)
    b = np.array(b.detach().numpy(), requires_grad=False)

    # H = a*qml.Projector(basis_state=[0], wires=the_wire) + b*qml.Projector(basis_state=[1], wires=the_wire)

    mat = np.array([[np.abs(a), 0], [0, np.abs(b)]])
    H = qml.Hermitian(mat, wires=the_wire)
    # obs = qml.Hamiltonian((1,), (H,))

    return H

########################################################################################################################


"""
FUNCTION USED TO CANCEL THE GLOBAL PHASE OF THE STATE
"""


def global_phase_operator(the_angle, the_wire: int = 0):
    """
    :param: the_angle: angle rotate the state with
    :param: the_wire: qubit on which the operator acts
    :return: None
    """
    qml.PhaseShift(the_angle, wires=the_wire)
    qml.PauliX(wires=the_wire)
    qml.PhaseShift(the_angle, wires=the_wire)
    qml.PauliX(wires=the_wire)

########################################################################################################################


dev1 = qml.device("default.qubit", wires=6)


@qml.qnode(dev1, interface='torch', diff_method="backprop")
def expect_value(the_G, the_n_layers, the_params, the_choice, massive: bool = False):
    """
    :param: the_G: graph representing the Feynman diagram
    :param: the_n_layers: number of layers
    :param: the_params: value of the parameters
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: expectation value of a diagonal, positive hermitian operator on the first qubit
    """

    the_circuit_params = the_params

    if the_choice == 'parametrized':
        parametric_qgnn(the_G, the_n_layers, the_circuit_params, massive=massive)
    elif the_choice == 'unparametrized':
        qgnn(the_G, the_n_layers, the_circuit_params, massive=massive)
    elif the_choice == 'fully_parametrized':
        fully_parametric_qgnn(the_G, the_n_layers, the_circuit_params, massive=massive)
    elif the_choice == 'fully_connected':
        fully_connected_qgnn(the_G, the_n_layers, the_circuit_params, massive=massive)
    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized', 'fully_parametrized' or 'fully_connected'")
        return 0

    # my_operator = qml.PauliZ(0)
    # my_operator = bhabha_operator(0, the_observable_params[0], the_observable_params[1]) #this operator is the one we want to define for the interference circuit
    # output = qml.expval(my_operator)

    # New way to define our customized observable
    output = qml.probs(wires=[0])

    return output


########################################################################################################################


# I'm working ong tree level QED with simple Feynman diagram, the number of vertices of the diagram is fixed and equal
# to 6, then I add one more qubit as a control qubit

"""
Circuit that extract the squared total matrix element
of Bhabha scattering (generalizable to other scattering processes)

"""
dev2 = qml.device("default.qubit", wires=7)


@qml.qnode(dev2, interface='torch', diff_method="adjoint")
def total_matrix_circuit(the_s_channel, the_s_params, the_s_phase, the_t_channel, the_t_params, the_t_phase,
                         the_layers, the_choice: str = 'parametrized', massive: bool = True):
    """
    Circuit that extract the squared total matrix element of 2 Feynman Diagrams
    :param: the_s_channel: graph representing the s-channel diagram
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_s_phase: global phase of the s-channel
    :param: the_t_channel: graph representing the t-channel diagram
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_t_phase: global phase of the t-channel
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: expectation value of a composite operator that is the prediction of the matrix element squared
    """

    the_s_observable = the_s_params[-2:]
    the_s_params = the_s_params[:-2]
    the_t_observable = the_t_params[-2:]
    the_t_params = the_t_params[:-2]

    qml.Hadamard(wires=6)

    if the_choice == 'parametrized':
        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    elif the_choice == 'unparametrized':
        qml.ctrl(qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    elif the_choice == 'fully_parametrized':
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers, the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers, the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    qml.Hadamard(wires=6)

    alpha = torch.abs(the_s_observable[0]*the_t_observable[0])
    beta = torch.abs(the_s_observable[1]*the_t_observable[1])
    my_operator = bhabha_operator(0, torch.sqrt(alpha), torch.sqrt(beta))

    print(qml.expval(my_operator))

    return qml.expval(my_operator @ qml.Projector(basis_state=[0], wires=6))


"""
Circuit that extract the interference term of Bhabha scattering (generalizable to other 
scattering processes)
"""
dev3 = qml.device("default.qubit", wires=7)


@qml.qnode(dev3, interface='torch', diff_method='adjoint')
def interference_circuit(the_s_channel, the_s_params, the_s_phase, the_t_channel, the_t_params, the_t_phase,
                         the_layers, the_choice: str = 'parametrized', massive: bool = False):
    """
    Circuit that extract the interference term of 2 Feynman Diagrams
    :param: the_s_channel: graph representing the s-channel diagram
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_s_phase: global phase of the s-channel
    :param: the_t_channel: graph representing the t-channel diagram
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_t_phase: global phase of the t-channel
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: expectation value of a composite operator that is the prediction of the interference
    """

    the_s_observable = the_s_params[-2:]
    the_s_params = the_s_params[:-2]
    the_t_observable = the_t_params[-2:]
    the_t_params = the_t_params[:-2]

    qml.Hadamard(wires=6)

    if the_choice == 'parametrized':

        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers[0], the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers[1], the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    elif the_choice == 'unparametrized':
        qml.ctrl(qgnn, control=6, control_values=1)(the_s_channel, the_layers[0], the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(qgnn, control=6, control_values=1)(the_t_channel, the_layers[1], the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    elif the_choice == 'fully_parametrized':
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_s_channel, the_layers[0], the_s_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_s_phase)
        qml.PauliX(wires=6)
        qml.ctrl(fully_parametric_qgnn, control=6, control_values=1)(the_t_channel, the_layers[1], the_t_params, massive=massive)
        qml.ctrl(global_phase_operator, control=6, control_values=1)((-1)*the_t_phase)

    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    qml.Hadamard(wires=6)

    alpha = torch.abs(the_s_observable[0] * the_t_observable[0])
    alpha.requires_grad = False
    beta = torch.abs(the_s_observable[1] * the_t_observable[1])
    beta.requires_grad = False
    my_operator = bhabha_operator(0, torch.sqrt(alpha), torch.sqrt(beta))

    return qml.expval(my_operator @ qml.PauliZ(wires=6))


"""
Quantum phase estimation circuit for extrapolating the global phase of the circuit 
"""
dev4 = qml.device("default.qubit", wires=6)


@qml.qnode(dev4, interface='torch')
def phase_estimation(the_channel, the_final_params, the_n_layers,  the_choice, massive: bool = False):
    """
    Circuit that extract the interference term of 2 Feynman Diagrams
    :param: the_channel: graph representing the s-channel diagram
    :param: the_final_params: value of the final parameters after training
    :param: the_n_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: global phase of the circuit
    """

    the_circuit_params = the_final_params[:-2]

    if the_choice == 'parametrized':

        parametric_qgnn(the_channel, the_n_layers, the_circuit_params, massive=massive)

    elif the_choice == 'unparametrized':
        qgnn(the_channel, the_n_layers, the_circuit_params, massive=massive)

    elif the_choice == 'fully_parametrized':
        fully_parametric_qgnn(the_channel, the_n_layers, the_circuit_params, massive=massive)

    else:
        print("Error, the_choice must be either 'parametrized', 'unparametrized' or 'fully_parametrized'")
        return 0

    return qml.state()