# DeepRL-SpaceInvaders

This project implements a Deep Reinforcement Learning (DRL) agent to play the classic Atari game Space Invaders using the DQN (Deep Q-Network) algorithm. The project includes a Jupyter notebook (`space_invaders.ipynb`) that walks through the process of training and testing the agent.

## Getting Started

### Prerequisites

Before you start, ensure you have the following installed:

- Conda
- Python 3.7

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/billy-enrizky/DeepRL-SpaceInvaders.git
    cd DeepRL-SpaceInvaders
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda create --name space_invaders_env python=3.7
    conda activate space_invaders_env
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Download the Weights for 10K and 10M Training Steps
Download the weight here: [link](https://drive.google.com/file/d/1TgfGittIQC2KhNbut2l4NSASUr0swe1u/edit)

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2. **Open `space_invaders.ipynb`** and follow the instructions to train and test the DRL agent.

### Running the Script

You can also run the script directly to test the pre-trained model:

```python
import gym
import random
import pprint
import ale_py
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions)

dqn.compile(Adam(lr=1e-4))

dqn.load_weights('SavedWeights/1m/dqn_weights.h5f')

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))
```

## GitHub Pages

You can also view the project walkthrough on the GitHub Pages site: DeepRL-SpaceInvaders.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
