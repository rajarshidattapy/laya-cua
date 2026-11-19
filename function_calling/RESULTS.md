# Function-calling routing with Laya (glaive-function-calling-v2)

This is a routing task, not computer use. Laya scores typed options in one forward pass. Here it decides (1) should the assistant call a function now, and (2) which listed function. It does not write function arguments.

Setup: glaiveai/glaive-function-calling-v2, split by conversation (train: conversations 0-4499, dev: 4500-5199, test: 5200-6499). Rows are assistant turns, only conversations with 1-8 distinct functions. 2 epochs from base convaiinnovations/laya on a free Colab T4, RLCD loss, one run, one seed. Script: `laya_fc.py`.

Rows: train 4500, dev 1454, test 2574 (906 are call turns). Call rate 0.352 on test.

## Test split

| Metric | Base Laya | Fine-tuned | Always "no call" | Random pick |
|---|---|---|---|---|
| tool_call accuracy (n=2574) | 0.3042 | 0.9545 | 0.648 | n/a |
| tool_choice accuracy (n=906) | 0.9989 | 1.000 | n/a | 0.8786 |

Dev split: tool_call 0.2978 (base) vs 0.9512 (fine-tuned); tool_choice 0.9961 vs 1.000.

## Read this before quoting it

- The real gain is the call / no-call decision: 0.30 to 0.95, and above the 0.648 always-no-call baseline.
- tool_choice is already near the ceiling for base Laya. Most prompts list only one or two functions, so random guessing scores 0.88. The gain over base is not meaningful.
- Single run, single seed, no confidence intervals. Training loss moved from 0.547 (epoch 1) to 0.540 (epoch 2).
- The fine-tuned checkpoint was not saved (the Colab runtime was lost), so there are no published weights for this experiment yet.
