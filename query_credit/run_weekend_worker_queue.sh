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
report_root=$("$python" - "$config" "$profile" <<'PY'
import pathlib, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print((pathlib.Path(cfg["work_dir"]).resolve() / sys.argv[2] / "reports"))
PY
)
while IFS=$'\t' read -r first second; do
  [[ -z "$first" ]] && continue
  output=''
  case "$stage" in
    ig)
      printf -v shard '%02d' "$first"
      output="$report_root/ig/ig_shard_${shard}.csv"
      ;;
    gradient)
      output="$report_root/gradient/gradient_seed_${first}.csv"
      ;;
    micro)
      safe_method=${second//\//-}
      output="$report_root/micro/micro_seed_${first}_${safe_method}.csv"
      ;;
    *)
      echo "Unknown worker stage: $stage" >&2
      exit 2
      ;;
  esac
  if [[ -s "$output" ]]; then
    echo "[weekend] Resume: skipping completed $stage job $first $second"
    continue
  fi
  case "$stage" in
    ig)
      "$python" -m stackpilot.query_credit_weekend_ig \
        --config "$config" --profile "$profile" \
        --shard-index "$first" --shard-count "$second"
      job_status=$?
      ;;
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
  esac
  if (( job_status != 0 )); then
    printf '%s\t%s\t%s\n' "$first" "$second" "$job_status" >> "$failure_log"
    status=1
  fi
done < "$queue"
exit "$status"
