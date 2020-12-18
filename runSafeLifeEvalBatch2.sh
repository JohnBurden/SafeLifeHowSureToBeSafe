#!/bin/sh
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    start-training TrainDQNAppendStillConfidenceEntropy0.75 --port 6969 --algo confidenceDQN --env-type append-still-all --run-benchmark --which-benchmark "random/append-still-level$i-benchmark.yaml"
done
