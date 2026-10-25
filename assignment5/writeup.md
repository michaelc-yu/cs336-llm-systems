
# Problem (prompting_baselines)

(a)
Question-only:
* Format=1, Correctness=1: 2
* Format=1, Correctness=0: 203
* Format=0, Correctness=0: 1114

R1-zero:
* Format=1, Correctness=1: 0
* Format=1, Correctness=0: 795
* Format=0, Correctness=0: 524

R1-zero 3-shot:
* Format=1, Correctness=1: 224
* Format=1, Correctness=0: 1050
* Format=0, Correctness=0: 45

Observed 10 examples of format reward 1 and correctness reward 0, and 1/10 of them the model output was actually correct but just not parsed correctly.

Example:
* Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
* Ground truth: 540
* Completion: He walks at a speed of 5 m/sec. </think> <answer> He would run 3 * 3 = 9 sprints in a week, totaling 9 * 60 = 540 meters. </answer>

Observed 10 examples of format reward 0 and correctness reward 0, and similarly 1/10 of them the model output was correct but not parsed correctly.

Example:
* Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
* Ground truth: 3
* Completion: A robe takes 2 bolts of blue fiber and half that much white fiber. So, a robe takes 2 + (2 / 2) = 3 bolts of fiber in total.</think> <answer> It takes 3 bolts of fiber in total.</answer>

(b) It is not enough to just provide the question to the model (question_only prompt). Many times, the model simply continues the question, and outputs "Remember students don’t have to use any variable to compute in their answers" or "What's the final answer?".

For r1_zero prompts, many times the model doesn't actually think about the question presented to it, but instead some other random GSM8K-style problem. 

The model does the best for r1_zero_three_shot prompts. The model's completion and thinking step is now grounded in the actual question since it was presented with few-shot examples.



# Problem (grpo_experiments_standard_on_policy)

(a) See grpo_experiments.py

(b) Ran it for 50 steps using r1 zero three shot prompt. The r1 zero prompt never seems to get any correct answers so all rewards are always 0; the model is unable to learn.

![50 steps val reward](runs/50steps_val_reward.png)

Rollouts also look reasonable.

Example of incorrect rollout:

* Q: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
* A: Janet’s ducks lay 16 eggs per day. She eats 3 eggs for breakfast, so that leaves 16 - 3 = 13 eggs. She bakes muffins for her friends with 4 eggs each, which means she bakes 4 * 2 = 8 muffins. So she has 13 - 8 = 5 eggs left to sell. At $2 per duck egg, she makes 5 * 2 = $10 every day at the farmers' market. </think> <answer> 10 </answer>

Example of correct rollout:

* Q: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
* A: A robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2 / 2 = 1 bolt. So in total, it takes 2 + 1 = 3 bolts. </think> <answer> 3 </answer>

(c)
Batch loss.
![batch loss across 4 seeds](runs/50steps_4seeds_batch_loss.png)

Gradient norm increasing from ~0.55 to ~0.65 with occasional spikes to ~1 indicates that the magnitude of the policy updates is gradually increasing over time. Some rollout batches produce unusually strong learning signals.
![grad norm across 4 seeds](runs/50steps_4seeds_grad_norm.png)

Decreases from 0.7 to 0.2. The model becomes more deterministic in its predictions / responses.
![token entropy across 4 seeds](runs/50steps_4seeds_token_entropy.png)

Increase from 0.2 to 0.5, then seems to oscillate. Noisy since it's per-step batches.
![train mean reward across 4 seeds](runs/50steps_4seeds_train_mean_reward.png)

Very noisy.
![train mean format reward across 4 seeds](runs/50steps_4seeds_train_mean_format_reward.png)

Monotonic increase for val mean reward.
![val mean reward across 4 seeds](runs/50steps_4seeds_val_mean_reward.png)

Seems to be a slight decreasing trend for val mean format reward.
![val mean format reward across 4 seeds](runs/50steps_4seeds_val_mean_format_reward.png)

Average response length increases across all 4 seeds. Perhaps the longer responses are causing higher chances of formatting incorrectly?
![val avg response length across 4 seeds](runs/50steps_4seeds_val_avg_response_length.png)


Looking at a few examples of rollouts before and after training, I see that before training the model averages ~2/8 samples correct, but at the end of training the model averages ~6/8 samples correct. The reasoning also became much more coherent, with the model improving from obvious reasoning failures to systematically decomposing the problem.


# Problem (grpo_learning_rate)

Sweeped over 1e-6, 1e-5, 3e-5, 1e-4. Learning rate of 3e-5 works the best. 1e-6 is too small to do meaningful learning and 1e-4 is too large and diverges.

![learning rate sweep](runs/50steps_learningrate_sweep.png)


# Problem (grpo_prompt_ablation)

The r1 zero three shot prompt performs by far the best. Based on the prompting baselines study, the model gets almost no correctness reward in both the question only prompt and r1-zero prompt scenarios. The model did improve for the r1 zero prompt at the end though (train and val mean_reward of ~0.06), which might mean that if trained for more iterations, the correct responses could start getting reinforced and the rewards might keep climbing, but need to verify by increasing number of iterations (e.g. 200 instead of 50).

![prompt ablation](runs/50steps_prompt_ablation.png)


# Problem (think_about_length_normalization)

Sequence normalization gives each response equal weight regardless of response length. But this means that tokens in shorter responses have more weight than tokens in longer responses which could bias training toward shorter responses. Constant normalization gives each token equal weight, but longer responses then contribute more to the overall gradient and can dominate training. In our run, the average validation response length increased from ~108 to ~130+ tokens, so sequence normalization keeps the influence of each response more consistent as responses get longer, while constant normalization would give those longer responses more influence. For our GRPO setup, it seems that sequence normalization is more appropriate since the rewards and advantages are assigned at the response level.


# Problem (think_about_rft)

RFT reinforces every response with a positive reward, whereas Dr. GRPO uses the reward relative to the group mean. If all responses in a group are correct, Dr. GRPO assigns zero advantage to every response and produces no update, while RFT reinforces all of the correct responses. If all responses are incorrect, both objectives produce no update. Thus, Dr. GRPO is more focused on improving responses relative to other responses in the same group, while RFT can reinforce successful responses even when the entire group is successful.


# Problem (think_about_advantage_normalization)

With no normalization, we preserve the magnitude of the reward differences but this can lead to very different gradient scales across groups. Normalizing by the group mean makes the advantages relative to the average reward, but this seems problematic when the group mean is very small, as we would be dividing the advantages by a very small number. Normalizing by the group standard deviation seems more useful when a group contains both correct and incorrect responses, since it gives a more consistent scale across groups, although it can give no signal if every response in the group has same reward. I would expect std normalization to work better especially for mixed reward groups.


# Problem (grpo_experiments_variants_on_policy)

![train mean reward for all runs](runs/onpolicyvariants-mean_reward.png)

![val mean reward for all runs](runs/onpolicyvariants-val_mean_reward.png)

The mean validation rewards:
- GRPO_constant: 0.41125
- Dr_GRPO: 0.3725
- RFT: 0.40125
- MaxRL: 0.3175

Highest variance: MaxRL
Lowest variance: RFT

Mean validation reward excluding the 2 outliers:
- Dr_GRPO 0.4233
- MaxRL 0.40

There seems to be two runs (once for MaxRL and once for Dr_GRPO) that experiences a "collapse". The mean_token_entropy plot also shows only these two runs' token entropies blow up which can come from steps that are too large.

Without the collapsed seeds, Dr_GRPO performs the best.


# Problem (think_about_importance_reweighting)

A major bottleneck in LLM post-training is the high cost of generating rollouts compared with performing additional optimization steps on existing data. Off-policy RL allows to overcome this bottleneck by generating a batch of text once, and then running multiple optimization steps over that data instead of a single gradient step. However, as the policy changes, the distribution of the previously generated data increasingly diverges from the current policy. Important reweighting is a strategy that helps against this bias by weighting samples according to the ratio between their probabilities under the current policy and the behavior policy that generated them.

- For no reweighting, the bias would be high but the variance is low.
- PPO/GRPO-style clipped token-level importance reweighting would have low bias but would have high variance since tracking token-level adjustments at each individual step could introduce instability and noisy optimizations.
- GSPO instead applies importance reweighting at the sequence level rather than for each individual token. This helps reduce the impact of noisy token-level ratios and makes training more stable. However it loses some of the fine-grained correction obtained from token-level reweighting, and clipping still introduces some bias. So it is more of a middle ground.

