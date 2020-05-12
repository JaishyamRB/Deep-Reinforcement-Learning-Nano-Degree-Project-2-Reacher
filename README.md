# Udacity Deep Reinforcement Learning Nanodegree Project 2: Continuous Control

## Project description

In this project, an agent learns to control a double-jointed arm to follow the target locations (Reacher environment).
A reward of +0.1 is provided for each step that the agent's hand is in the target location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints.
Every entry in the action vector is a number between -1 and 1.
The task is episodic, and in order to solve the environment, an agent must get an average score of +30 over 100 consecutive episodes.
Here are the Unity details of the environment:
```
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```
## Getting Started

1. Before getting into the project there are certain dependencies to be met. Make sure you have [python 3.6]( https://www.python.org/downloads/release/python-3610/) installed and virtual environment.

2. Download the environment from one of the links below.
You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

3. Download or clone this repo and run these commands in your terminal:

```
pip install requirements.txt
pip -q install ./python
```

## Instructions

To run the agent open [Solution.ipynb](Solution.ipynb)

Description of the implementation is provided in [Report.md](Report.md). 
For technical details see the code.

Actor and critic model weights are stored in [actor.pth](actor.pth) and [critic.pth](critic.pth), respectively.
