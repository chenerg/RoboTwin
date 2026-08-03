#!/usr/bin/env bash

# Collect demonstrations for the 30 newly proposed RoboTwin tasks.
# collect_data.py already resumes seed search and HDF5 replay, so rerunning this
# script safely continues each task from its existing output directory.

set -u

usage() {
    cat <<'EOF'
Usage:
  bash collect_new_tasks_data.sh <task_config> <gpu_id> [options]

Options:
  --dry-run            Print collection commands without executing them.
  --skip-missing       Skip tasks whose policy/instruction/eval config is absent.
  --continue-on-error  Continue with later tasks after one collection fails.
  -h, --help           Show this help message.

Examples:
  bash collect_new_tasks_data.sh demo_clean 0
  bash collect_new_tasks_data.sh demo_randomized 1 --continue-on-error
  bash collect_new_tasks_data.sh demo_clean 0 --skip-missing --dry-run
EOF
}

if [[ $# -eq 1 && ($1 == -h || $1 == --help) ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

task_config=${1%.yml}
gpu_id=$2
shift 2

dry_run=false
skip_missing=false
continue_on_error=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=true
            ;;
        --skip-missing)
            skip_missing=true
            ;;
        --continue-on-error)
            continue_on_error=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
collector="$repo_root/collect_data.sh"
config_path="$repo_root/task_config/$task_config.yml"
eval_config="$repo_root/task_config/_eval_step_limit.yml"

if [[ ! $task_config =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid task config: $task_config" >&2
    exit 2
fi
if [[ ! -f $config_path ]]; then
    echo "Task config not found: $config_path" >&2
    exit 2
fi
if [[ ! -f $collector ]]; then
    echo "Collector not found: $collector" >&2
    exit 2
fi
if [[ ! -f $eval_config ]]; then
    echo "Eval step-limit config not found: $eval_config" >&2
    exit 2
fi

tasks=(
    # Implemented procedural block tasks.
    place_blue_block_green_pad
    place_green_block_yellow_pad
    place_orange_block_purple_pad
    place_purple_block_orange_pad
    place_red_block_blue_pad
    place_yellow_block_red_pad
    place_red_block_left_of_blue_block
    place_red_block_right_of_blue_block
    place_red_block_in_front_of_blue_block
    place_red_block_behind_blue_block
    stack_red_block_on_blue_block
    stack_blue_block_on_red_block
    stack_green_block_on_yellow_block
    stack_purple_block_on_orange_block
    rank_blocks_blue_green_red
    rank_blocks_purple_blue_green
    rank_blocks_yellow_orange_red
    place_red_blue_blocks_opposite_pads
    place_green_yellow_blocks_matching_pads
    place_orange_purple_blocks_opposite_pads

    # Proposed heterogeneous tasks. Their policies must exist before collection.
    close_laptop
    open_then_close_cabinet_drawer
    insert_markpen_into_pencup
    strike_gong_with_mallet
    wipe_mini_chalkboard
    pour_beads_between_bowls
    push_toycar_to_parking_zone
    handover_dumbbell
    balance_globe_on_displaystand
    weigh_then_remove_object
)

missing_tasks=()

for task in "${tasks[@]}"; do
    missing_parts=()
    policy_path="$repo_root/envs/$task.py"
    instruction_path="$repo_root/description/task_instruction/$task.json"

    if [[ ! -f $policy_path ]]; then
        missing_parts+=(policy)
    elif ! grep -Eq "^class[[:space:]]+$task([[:space:]]|\()" "$policy_path"; then
        missing_parts+=(same-named-class)
    fi
    [[ -f $instruction_path ]] || missing_parts+=(instruction)
    grep -Eq "^$task:[[:space:]]*[0-9]+[[:space:]]*$" "$eval_config" || missing_parts+=(eval-step-limit)

    if [[ ${#missing_parts[@]} -gt 0 ]]; then
        missing_tasks+=("$task")
        printf 'Missing %-39s %s\n' "$task" "${missing_parts[*]}" >&2
    fi
done

if [[ ${#missing_tasks[@]} -gt 0 && $skip_missing == false ]]; then
    echo >&2
    echo "Preflight failed: ${#missing_tasks[@]} of ${#tasks[@]} tasks are incomplete." >&2
    echo "Implement their policy/instruction/eval entries, or pass --skip-missing to collect only ready tasks." >&2
    exit 2
fi

is_missing() {
    local candidate=$1
    local missing
    [[ ${#missing_tasks[@]} -eq 0 ]] && return 1
    for missing in "${missing_tasks[@]}"; do
        [[ $candidate == "$missing" ]] && return 0
    done
    return 1
}

ready_count=$((${#tasks[@]} - ${#missing_tasks[@]}))
echo "Task configuration : $task_config"
echo "GPU id             : $gpu_id"
echo "Ready tasks        : $ready_count/${#tasks[@]}"
echo "Dry run            : $dry_run"
echo "Continue on error  : $continue_on_error"
echo

completed=0
failed=0
skipped=0
task_index=0

for task in "${tasks[@]}"; do
    task_index=$((task_index + 1))

    if is_missing "$task"; then
        skipped=$((skipped + 1))
        printf '[%02d/%02d] SKIP %s (incomplete task)\n' "$task_index" "${#tasks[@]}" "$task"
        continue
    fi

    printf '[%02d/%02d] START %s\n' "$task_index" "${#tasks[@]}" "$task"
    if [[ $dry_run == true ]]; then
        printf '  bash %q %q %q %q\n' "$collector" "$task" "$task_config" "$gpu_id"
        completed=$((completed + 1))
        continue
    fi

    if bash "$collector" "$task" "$task_config" "$gpu_id"; then
        completed=$((completed + 1))
        printf '[%02d/%02d] DONE  %s\n' "$task_index" "${#tasks[@]}" "$task"
    else
        collect_status=$?
        failed=$((failed + 1))
        printf '[%02d/%02d] FAIL  %s (exit=%d)\n' "$task_index" "${#tasks[@]}" "$task" "$collect_status" >&2
        if [[ $continue_on_error == false ]]; then
            echo "Stopped after the first failure. Rerun the same command to resume." >&2
            exit "$collect_status"
        fi
    fi
done

echo
echo "Batch summary: completed=$completed failed=$failed skipped=$skipped total=${#tasks[@]}"

if [[ $failed -gt 0 ]]; then
    exit 1
fi
