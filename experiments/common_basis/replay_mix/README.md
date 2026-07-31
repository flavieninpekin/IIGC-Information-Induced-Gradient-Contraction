# Experiment 3: DQN replay-buffer relation-mix ablation

Goal: test the "TD reconciliation" mechanism directly. Train/measure DQN on
replay data from a single relation vs an increasingly uniform mixture of
relations. Prediction: kappa on the TD-loss field rises with mixing fraction.

This isolates the aggregate-consistency of the TD target as the cause of high
value-based kappa, and pairs with the behavioral check (high kappa + low
reward = "aggregated but unactionable").
