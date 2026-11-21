# How well do transcoders predict GPT-2’s response to an activation edit?

A transcoder approximates an MLP by encoding its input into sparse features and decoding those features into a predicted output. We compare a learned transcoder with one whose encoder directions are frozen at initialization. When the two reconstruct ordinary MLP outputs about equally well, which one better predicts how the output changes **after an input edit**?

On held-out WikiText-103 articles, the answer changed with training time. The frozen encoder predicted edits more accurately after 2,000 updates. By 20,000 updates, the learned transcoder had a small average advantage. This reversal appeared in all three initialization seeds for the GPT-2 block we tested.

## Guide to the notebooks

| File | What to look for |
|---|---|
| [01_02_intervention_checks.ipynb](./01_02_intervention_checks.ipynb) | Starts with forward hooks and an identity patch. An exact replacement and a shortcut can agree on clean examples yet respond differently when one input changes. ReLU and redundant-pathway examples check the evaluator against known answers. |
| [03_transcoder_training_pilot.ipynb](./03_transcoder_training_pilot.ipynb) | Trains transcoders on scalar and vector toy functions. Compares learned, frozen-encoder and frozen-decoder variants, then investigates the frozen encoder’s poor fit. On the vector toy, a least-squares decoder refit recovers the tested responses to numerical precision. |
| [04_gpt2_transcoder_pilot.ipynb](./04_gpt2_transcoder_pilot.ipynb) | Caches GPT-2 block 5 activations, compares sparse replacements, selects a reference width by validation reconstruction error, and evaluates saved checkpoints across training budgets and seeds. The final cells score held-out documents and compute bootstrap intervals. |
| Experiment report | Summarizes the GPT-2 comparison, including width selection, seed-level results, the held-out protocol and uncertainty estimates. |

## How the GPT-2 comparison works

For each test edit, we blend an MLP input halfway toward a donor activation from the same token position in a different article. We evaluate the original MLP before and after the edit, then ask each transcoder to predict that output difference. The response error, **E**, is the sum of squared prediction errors divided by the sum of squared true output changes. Lower is better.

At TopK = 128, we selected the learned width on validation data before scoring interventions. All three seeds selected 192 learned features against the 3,072-feature frozen reference. Both variants received the same ridge decoder refit at each checkpoint. All reported pairs stayed within the 5% relative tolerance on normalized reconstruction error.

## Held-out results

The test panel contains 236 edits across 59 root articles. The table reports the mean across three seeds. Positive differences favor the learned transcoder.

| Training updates | Response error difference (frozen − learned) | 95% paired article bootstrap interval |
|---:|---:|---:|
| 2,000 | −0.153 | [−0.163, −0.143] |
| 4,000 | −0.117 | [−0.128, −0.107] |
| 8,000 | −0.062 | [−0.074, −0.050] |
| 20,000 | +0.020 | [+0.005, +0.034] |

The final gap falls below the report’s proposed threshold of +0.05 for a meaningful learned advantage. At that checkpoint, the learned model also had about 3.2% lower reconstruction error and used about 125 active features versus 128 for the frozen encoder. Those differences limit how much of the final gap we can attribute to learning the encoder directions. The intervals describe variation across sampled root articles for these fitted models and donor assignments.

## Running the experiments

Start with the intervention checks to see how the response metric behaves on functions with known answers. The training notebooks were developed in Google Colab. Before running training or restoration cells, review their Drive paths and checkpoint settings. The GPT-2 analysis uses caches and saved runs under `MyDrive/transcoder_pilot/gpt2_block5`. To reproduce the comparison from scratch, generate the activation caches and training checkpoints before running the held-out scoring cells.

The toy training notebook uses the [transcoder_circuits implementation](https://github.com/jacobdunefsky/transcoder_circuits).
