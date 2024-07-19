#Implements the RewardModelScores and PropositionProbabilities classes using cached data from file

from base_classes import RewardModelScores, FeaturesValues
from typing import List, Dict
from utils import read_jsonl, read_yaml, to_string

CONFIG_FILE_PATH = 'config/proposition_prompts.yaml' #not used in this example since everything is already cached, but it generated the cached data
CACHED_FEATURES = 'data/weight_fitting_data/prop_probs/{split}.jsonl'
CACHED_REWARD_MODEL_SCORES = 'data/weight_fitting_data/rewards/{rm_size}/{split}.jsonl'
RESPONSE_TYPES = ['Hard Refuse', 'Comply', 'Safe Refuse 1', 'Safe Refuse 2']



class CachedRewardModelScores(RewardModelScores):
    def __init__(self, rm_size="large"):
        train_data_raw = read_jsonl(CACHED_REWARD_MODEL_SCORES.format(rm_size=rm_size, split='train'))
        test_data_raw = read_jsonl(CACHED_REWARD_MODEL_SCORES.format(rm_size=rm_size, split='test'))

        #items are hashed by (prompt, completion) pairs
        self.train_data = {tuple(d['prompt_completion']): d['rm_score'] for d in train_data_raw}
        self.test_data = {tuple(d['prompt_completion']): d['rm_score'] for d in test_data_raw}
    
    def get_reward_model_scores(self, prompt: List[Dict[str, str]], completions: List[List[Dict[str, str]]]) -> List[float]:
        completion_rewards = []
        for idx_c, completion in enumerate(completions):
            lookup_hash = (to_string(prompt), to_string(completion))
            if lookup_hash in self.train_data:
                completion_rewards.append(self.train_data[lookup_hash])
            elif lookup_hash in self.test_data:
                completion_rewards.append(self.test_data[lookup_hash])
            else:
                raise ValueError(f"No cached reward model score found for prompt: {prompt}, completion: {idx_c}, {completion}")

        return completion_rewards

class CachedFeaturesValues(FeaturesValues):
    def __init__(self):
        train_data_raw = read_jsonl(CACHED_FEATURES.format(split='train'))
        test_data_raw = read_jsonl(CACHED_FEATURES.format(split='test'))

        #items are hashed by (prompt, completion) pairs
        self.train_data = {tuple(d['prompt_completion']): d['features'] for d in train_data_raw}
        self.test_data = {tuple(d['prompt_completion']): d['features'] for d in test_data_raw}

        #not used in this example since everything is already cached, but it generated the cached data
        self.features_config = read_yaml(CONFIG_FILE_PATH)

    def get_features(self, prompt: List[Dict[str, str]], completions: List[List[Dict[str, str]]]) -> List[float]:
        completion_features = []
        for idx_c, completion in enumerate(completions):
            lookup_hash = (to_string(prompt), to_string(completion))
            if lookup_hash in self.train_data:
                completion_features.append(self.train_data[lookup_hash])
            elif lookup_hash in self.test_data:
                completion_features.append(self.test_data[lookup_hash])
            else:
                raise ValueError(f"No cached features found for prompt: {prompt}, completion: {idx_c}, {completion}")

        return completion_features
