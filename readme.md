# Safety RBR Gold Dataset and Weight Fitting Code

**Warning: Content may include language related to racism, erotic themes, self-harm, or other offensive material.**

This directory contains complementary code and data for the paper: Rule Based Rewards for Language Model Safety

It contains:

- Our Safety RBR gold dataset, the small set of human data we used in the this experiment. This dataset was used for prompt tuning and calculating the accuracy of prompt+LLM grader (ex. Table 13 in the paper.) The data lives in `data/rbr_gold_data/` and the notebook `analyze_RBR_gold_data.ipynb` gives further examples for loading the data.
- Our code for fitting the RBR weights (`rbr_weight_fitter.py`) along with an example `weight_fitting_example.ipynb` of usage and visualization.
- Some example synthetic data and reward model scores to demonstrate the usage of the weight fitting code (`data/weight_fitting_data/`)

A good starting place is the two notebooks we provide:

## Notebooks

1. Weight Fitting Example (`weight_fitting_example.ipynb`): This notebook provides an example of using the RBR weight fitting code given (`rbr_weight_fitter.py`) using the example synthetic data we provide. It demonstrates how to load data, fit weights, and visualize the results.
2. RBR Gold Data (`rbr_gold_data.ipynb`): This notebook covers the RBR Gold dataset, a small set of human-labelled data used for prompt tuning and prompt+LLM grader accuracy calculations. It includes example code for loading the data and some very basic statistical analysis.

## License

We are releasing this code and data under the MIT License.
