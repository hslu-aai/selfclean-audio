#!/bin/bash

echo "Running all baseline evaluations..."

echo "Running off-topic detection baselines..."
./scripts/baselines/get_off_topic_baselines.sh

echo "Running near duplicate detection baselines..."
./scripts/baselines/get_near_dup_baselines.sh

echo "Running label error detection baselines..."
./scripts/baselines/get_label_error_baselines.sh

echo "All baseline evaluations completed!"
