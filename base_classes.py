from abc import ABC, abstractmethod
from typing import List

class FeaturesValues(ABC):
    @abstractmethod
    def get_features(self, prompt: str, completions: List[str]) -> List[dict[str, float]]:
        #Takes in a prompt and a list of completions, returns a dictionary of feature names and their values
        pass

class RewardModelScores(ABC):
    @abstractmethod
    def get_reward_model_scores(self, prompt: str, completions: List[str]) -> List[float]:
        #Takes in a prompt and a list of completions, returns the reward model scores for each completion
        pass