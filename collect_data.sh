#!/usr/bin/env bash

set -u

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <task_name> <task_config> <gpu_id>" >&2
    exit 2
fi

task_name=$1
task_config=${2%.yml}
gpu_id=$3

if [[ ! $task_name =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid task name: $task_name" >&2
    exit 2
fi
if [[ ! $task_config =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid task config: $task_config" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

./script/.update_path.sh > /dev/null 2>&1

CUDA_VISIBLE_DEVICES=$gpu_id PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "$task_name" "$task_config"
collect_status=$?

cache_dir="data/$task_name/$task_config/.cache"
if [[ -d $cache_dir ]]; then
    rm -rf -- "$cache_dir"
fi

exit "$collect_status"
