

# Problem (chinchilla_isoflops)

(a) Optimal model size for a compute budget of $10^{23}$ FLOPs: 5e+10

Optimal model size for a compute budget of $10^{24}$ FLOPs: 1.27e+11

(b) Optimal dataset size for a compute budget of $10^{23}$ FLOPs: 3.37e+11

Optimal dataset size for a compute budget of $10^{24}$ FLOPs: 1.33e+12

Quick observation: these results show that D/N =~ 6-10x rather than 20x (as mentioned in the Chinchilla paper).


# Problem (scaling_laws)

Given 12 B200-hours for exploratory runs, I'd want to split the compute budget up so I have enough compute for multiple smaller experiments. I'd want to run multiple experiments across varying compute budgets. I would run more runs across smaller compute budgets since smaller compute runs are cheap so I can afford denser coverage there. I would run less larger compute runs and mainly use them to validate the scaling curve. For each compute budget, I'd try different combinations of parameters and dataset sizes (N and D). Given that C =~ 6ND, and Chinchilla finds that optimal split is roughly 20 tokens per parameters, I could sample from {6, 10, 15, 20, 25, 30} tokens per parameter. Then I'd query the training API with all these setups, and find the lowest final loss per compute budget size and fit a power law $y = a * x^b$ on them using scipy curve_fit method.

