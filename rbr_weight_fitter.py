from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

@dataclass
class RBR_ExampleWithMetadata:
    convo_prompt: str
    convo_completion: str
    response_type: str
    completion_label: str
    base_reward: Optional[float] = None
    features: Optional[Dict[str, float]] = field(default_factory=dict)

    def add_base_reward(self, base_reward: float):
        self.base_reward = base_reward

    def add_features(self, features: Dict[str, float]):
        self.features.update(features)

class RBRWeightFitter:
    def __init__(self, 
                 feature_names: List[str], 
                 examples: List[RBR_ExampleWithMetadata], 
                 orderings: Dict[str, List[str]], 
                 ignore_features: List[str] = [],
                 #optimization hyperparameters
                 train_data_frac: float = 0.95,
                 margin: float = 1,
                 lr: float = 1e-2,
                 wd: float = 0.0,
                 n_iters: int = 1000,
                 completion_label_margin: Dict[str, float] = {}):
        """
        Optimize a linear function for combining the RBR with the Reward Model by optimizing the hinge loss.

        Args:
            feature_names (list[str]): the list of feature names
            examples (list[RBR_ExampleWithMetadata]): the list of examples to optimize on
            orderings (dict[str, list[str]]): a dict of str -> list[str] where an example whose label is a key is considered better than any example whose label is in the corresponding values.
            ignore_features (list[str]): the list of features to ignore
            train_data_frac (float): the fraction of data to use for training
            margin (float): the margin for the hinge loss
            lr (float): the learning rate for the optimizer
            wd (float): the weight decay for the optimizer
            n_iters (int): the number of iterations to run the optimizer
            completion_label_margin (dict[str, float]): Pass in specific margins for completion labels to override the default margin. (ex may want to set a higher margin for "ideal" completions)
        """
        self.feature_names = feature_names
        self.examples = examples
        self.orderings = orderings
        self.ignore_features = ignore_features
        self.train_data_frac = train_data_frac
        self.margin = margin
        self.lr = lr
        self.wd = wd
        self.n_iters = n_iters
        self.completion_label_margin = completion_label_margin
        self.weights = torch.zeros(len(feature_names), requires_grad=True)
        self.metrics = {"frac_clipped": [], "loss": [], "valid_loss": []}

    def validate_examples(self):
        for ex in self.examples:
            assert ex.completion_label in self.orderings, "All examples must have a valid completion label."
            assert set(ex.features.keys()) == set(self.feature_names), "Must have values for all features."

    def get_orderings(self):
        def apply_ordering(i1, i2):
            ex1, ex2 = self.examples[i1], self.examples[i2]
            if not ex1.convo_prompt == ex2.convo_prompt:
                return None
            if ex2.completion_label in self.orderings[ex1.completion_label]:
                margin = self.completion_label_margin.get(ex1.completion_label, 1)
                return [i1, i2, margin]
            elif ex1.completion_label in self.orderings[ex2.completion_label]:
                margin = self.completion_label_margin.get(ex2.completion_label, 1)
                return [i2, i1, margin]
            return None

        pairs = list(combinations(range(len(self.examples)), 2))
        pairs = [apply_ordering(i1, i2) for i1, i2 in pairs]
        pairs = [p for p in pairs if p is not None]
        np.random.default_rng().shuffle(pairs)
        pairs = torch.tensor(pairs, dtype=torch.int64)
        return pairs

    def zero_some_feature_weights(self, grad):
        ignore_feature_idxs = [ft in self.ignore_features for ft in self.feature_names]
        grad[ignore_feature_idxs] = 0
        return grad

    def prepare_optimization(self):
        self.weights.register_hook(self.zero_some_feature_weights)
        opt = torch.optim.AdamW([self.weights], lr=self.lr, weight_decay=self.wd)
        base_rewards = torch.tensor([ex.base_reward for ex in self.examples])
        vals = torch.tensor([[ex.features[ft] for ft in self.feature_names] for ex in self.examples])
        assert vals.shape[0] == len(self.examples), f"vals shape[0] should be {len(self.examples)} but got {vals.shape[0]}"
        assert vals.shape[1] == len(self.feature_names), f"vals shape[1] should be {len(self.feature_names)} but got {vals.shape[1]}"
        return opt, base_rewards, vals

    def get_loss(self, idxs1, idxs2, margins, base_rewards, vals):
        total_scores = base_rewards + vals @ self.weights
        per_example_loss = F.relu(((self.margin * margins) + total_scores[idxs2]) - total_scores[idxs1])
        frac_clipped = 1 - (per_example_loss > 0).float().mean().item()
        loss = per_example_loss.mean()
        return loss, frac_clipped

    def fit_weights(self):
        """
        Returns:
            weights (dict[str, float]): the optimized weights for the features
            metrics (dict[str, list]): some metrics from the optimization
        """
        self.validate_examples()
        pairs = self.get_orderings()
        opt, base_rewards, vals = self.prepare_optimization()

        split_idxs = lambda P: (p.squeeze(-1) for p in P.split(1, dim=-1))
        train_points = int(len(pairs) * self.train_data_frac)
        train_idxs1, train_idxs2, train_margins = split_idxs(pairs[:train_points])
        valid_idxs1, valid_idxs2, valid_margins = split_idxs(pairs[train_points:])

        for _ in tqdm(range(self.n_iters)):
            opt.zero_grad()
            loss, frac_clipped = self.get_loss(train_idxs1, train_idxs2, train_margins, base_rewards, vals)
            loss.backward()
            opt.step()

            valid_loss, _ = self.get_loss(valid_idxs1, valid_idxs2, valid_margins, base_rewards, vals)
            self.metrics["frac_clipped"].append(frac_clipped)
            self.metrics["loss"].append(loss.item())
            self.metrics["valid_loss"].append(valid_loss.item())

        weights_dict = {ft: w for ft, w in zip(self.feature_names, self.weights.detach().numpy())}
        return weights_dict, self.metrics
