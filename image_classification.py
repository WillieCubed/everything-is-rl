"""
Observation: an image
Action: predict the type
"""
from typing import Literal, NamedTuple, Tuple, Union
from datasets import load_dataset
from gym import Env, Space
from gym.envs.registration import EnvSpec
import numpy as np

class PredictClass(NamedTuple):
    type: Literal['predict_class']
    value: int

class PredictImage(NamedTuple):
    type: Literal['predict_image']
    value: np.ndarray
    label: int

ActType = Union[PredictClass, PredictImage]

Observation = np.ndarray

class MNISTEnvironment(Env):
    # Set this in SOME subclasses
    metadata = {"render_modes": ["single_rgb_array", "human"]}
    render_mode = "human"  # define render_mode if your environment supports rendering
    reward_range = (-float("inf"), float("inf"))

    # Set these in ALL subclasses
    action_space: Space[ActType]
    observation_space: Space[Observation]

    def __init__(self, dataset, **kwargs) -> None:
        super().__init__()
        self._dataset = dataset
        self._current_index = 0
        self._current_sample = dataset[self._current_index]

    def step(self, action: ActType) -> Tuple[Observation, float, bool, dict]:
        last_image = self._current_sample['image']
        last_class = self._current_sample['label']
        predicted_class = action.label
        predicted_image = action.value
        reward = self._compute_reward(last_image, predicted_image, last_class, predicted_class)
        self._advance_sample()
        observation = self._current_sample
        image = observation['image']
        done = self._is_done()
        return image, reward, done, {}

    def _is_done(self):
        return self._current_index + 1 >= len(self._dataset)

    def _compute_reward(self, expected_image: np.ndarray, received_image: np.ndarray, expected_class: int, predicted_class: int) -> float:
        match = expected_class == predicted_class
        image_diff = expected_image - received_image
        normalized = np.sqrt(np.sum(np.square(image_diff)))
        # TODO: Choose less bogus reward function
        return 1 if match else -1

    def _advance_sample(self):
        next_index = self._current_index + 1
        self._current_index = next_index
        self._current_sample = self._dataset[next_index]


class Model:
    def __init__(self, image_shape) -> None:
        self._image_shape = image_shape
    
    def on_step(self, observation: np.ndarray, previous_reward: float):
        return np.random.rand(*self._image_shape)

    def update_model(self, observation) -> None:
        pass

def run(episodes=10):
    # Reward is accuracy points
    dataset = load_dataset('mnist')
    env = MNISTEnvironment()

    model = Model()

    episode = 0

    prediction = None
    total_reward = 0.0
    while episode < episodes:
        episode_reward = 0.0
        done = False
        # TODO: Insert threshold for *something*
        while not done:
            observation, episode_reward, done, info = env.step(prediction)
            prediction = model.on_step(observation, episode_reward)
        
        total_reward += episode_reward
    

if __name__ == "__main__":
    run()