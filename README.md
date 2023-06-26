
# **FDQLearn**

## Feynman Diagrams Quantum Learner:

The goal of the project is to create a quantum neural network that can efficiently compute the squared matrix element 
of a tree-level Feynman Diagram, given its topology and the necessary dynamical dofs.
This work is get inspired by the work: “Learning Feynman Diagrams using Graph Neural Networks”. In: arXiv preprint arXiv:2211.15348
(2022), Harrison Mitchell, Alexander Norcliffe, and Pietro Liò, where they define a 
graph neural network for high energy QED.

This project aims to be an original work that goes deeper in the theory
of quantum graph neural network and equivariant quantum machine learning
applied to QFT.
Using a QML approach seems natural to study QFT, because they follow
the same physical laws of quantum theory, and exploiting them can bring us to perform
straightforward computation in quantum regime that are unfeasible in classical regime,
in a sort of quantum advantage in terms of code complexity.

In this project I develop a quantum graph neural network because it is
the best way to encode the pictorial Feynman Diagrams, which contain
essential information on the kind of interaction and then on the output results.

Some of the future aspects of the project is to enlarge the kind of diagrams and 
interactions that can be studied with this approach and try to build a network that encodes the
symmetries of the interaction (both gauge and accidental symmetries) and study more 
nice computations that can't be done (either at all or in a feasible way) by 
classical machine learning.

## Installing the virtual environment (venv command)

Here are showed the steps to setup the virtual environment with
the right versions of the packages I've used; the version of Python is 3.9.13.

Create the environment:
1) $ conda env create -p ./.venv --file env.yml 

Activate the environment:

for Linux and MacOs:
1) $ source .venv/bin/activate

for Windows:
2) $ .venv/Scripts/activate

Install 'requirements.txt' and my library:
3) $ pip install -f requirements.txt
4)  $ pip install -e .


5) In repo's root run setup.py


## Regular use of the virtual environment (venv command)

for Linux and MacOs:
1) $ source .venv/bin/activate

for Windows:
1) $ .venv/Scripts/activate

run whatever cmd, es: $ python scripts/FDQLearn_main.py

## Dataset generation (venv command)
In the repo there are some existent datasets, but there's the original notebook
from the previous paper that create a dataset; the command to execute the jupyter notebook
(after activating the virtual environment) is:

$ ipython kernel install --user --name=.venv


## Installation of the virtual environment (conda)

Here are showed the steps to setup the virtual environment with
the right versions of the packages I've used; the version of Python is 3.9.13.

Create the environment:
1) $ conda env create -p ./.venv --file env.yml 

Activate the environment:
2) $ conda activate ./.venv

Install 'requirements.txt' and my library:
3) $ pip install -f requirements.txt
4)  $ pip install -e .

5) In repo's root run setup.py


## Regular use of the virtual environment (conda)

1) $ conda activate ./.venv

run whatever cmd, es: $ python scripts/FDQLearn_main.py


## Dataset generation (conda)

In the repo there are some existent datasets, but there's the original notebook
from the previous paper that create a dataset; the command to execute the jupyter notebook
(after activating the virtual environment) is:

$ conda run -p ./.venv jupyter nbconvert --to notebook --stdout --execute scripts/Dataset_builder.ipynb

