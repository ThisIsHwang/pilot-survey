#!/usr/bin/env bash
set -u
stage=$1
gpu=$2
queue=$3
python=$4
config=$5
profile=$6
export CUDA_VISIBLE_DEVICES=$gpu
status=0
failure_log="${queue%.txt}.failures.tsv"
: > "$failure_log"
while IFS=$'\t' read -r first second; do
  [[ -z "$first" ]] && continue
  case "$stage" in
    gradient)
      "$python" -m stackpilot.query_credit_weekend_gradient \
        --config "$config" --profile "$profile" --init-seed "$first"
      job_status=$?
      ;;
    micro)
      "$python" -m stackpilot.query_credit_weekend_micro \
        --config "$config" --profile "$profile" --run-seed "$first" --method "$second"
      job_status=$?
      ;;
    *)
      echo "Unknown worker stage: $stage" >&2
      exit 2
      ;;
  esac
  if (( job_status != 0 )); then
    printf '%s\t%s\t%s\n' "$first" "$second" "$job_status" >> "$failure_log"
    status=1
  fi
done < "$queue"
exit "$status"
