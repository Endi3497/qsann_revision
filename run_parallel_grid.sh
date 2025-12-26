#!/usr/bin/env bash
[ -z "${BASH_VERSION:-}" ] && { echo "Please run with bash: bash run_parallel_grid.sh"; exit 1; }
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

# Hyperparameter grids
vqc_layers=(2 1)
reuploading=(3 2 1)
attn_layers=(1 2 3)
hidden_dims=(__none__)
model_modules=(model)

# Dataset/device/image specs
specs=(
  "cuda:0 mnist 8"
  "cuda:1 mnist 28"
  "cuda:2 fmnist 28"
  "cuda:3 cifar10 32"
)

# Thresholds: acc auroc precision recall f1 auprc
thresholds() {
  case "$1" in
    mnist_8)    echo "0.7250 0.8852 0.5780 0.7069 0.6333 0.7620" ;;
    mnist_28)   echo "0.8375 0.9700 0.8447 0.8310 0.8285 0.8488" ;;
    fmnist_28)  echo "0.7500 0.9631 0.7231 0.7325 0.7076 0.7847" ;;
    cifar10_32) echo "0.2250 0.6446 0.2341 0.2171 0.2016 0.2481" ;;
    *) echo "0 0 0 0 0 0" ;;
  esac
}

extract_metrics() {
  local log="$1"
  local line
  line=$(grep -E "Test -> acc:" "$log" | tail -n1 || true)
  if [ -z "$line" ]; then
    echo ""; return 1
  fi
  # Expect fields: acc: X auroc: Y precision: Z recall: W f1: V auprc: U
  awk '{for(i=1;i<=NF;i++){if($i=="acc:") a=$(i+1); if($i=="auroc:") b=$(i+1); if($i=="precision:") c=$(i+1); if($i=="recall:") d=$(i+1); if($i=="f1:") e=$(i+1); if($i=="auprc:") f=$(i+1);} printf "%s %s %s %s %s %s",a,b,c,d,e,f;}' <<<"$line"
}

meets_thresholds() {
  local key="$1"; shift
  local metrics=($*)
  local thresh=($(thresholds "$key"))
  if [ ${#metrics[@]} -lt 2 ]; then return 1; fi
  # Only check accuracy (idx 0) and auroc (idx 1)
  for i in 0 1; do
    if ! awk -v m="${metrics[$i]}" -v t="${thresh[$i]}" 'BEGIN{exit !(m>t)}'; then
      return 1
    fi
  done
  return 0
}

for r in "${reuploading[@]}"; do
  for v in "${vqc_layers[@]}"; do
    for al in "${attn_layers[@]}"; do
      for hd in "${hidden_dims[@]}"; do
        echo "=== Trying combo: vqc_layers=$v reuploading=$r attn_layers=$al hidden_dim=$hd (models in parallel) ==="
        pids=()
        tags=()
        for m in "${model_modules[@]}"; do
          for spec in "${specs[@]}"; do
            read -r device ds img <<<"$spec"
            run_name="grid-${ds}-img${img}-m${m}-v${v}-r${r}-al${al}-hd${hd}"
            tags+=("$run_name|$ds|$img")
            DEVICE="$device" DATASET_CHOICE="$ds" IMAGE_SIZE="$img" VQC_LAYERS="$v" REUPLOADING="$r" ATTN_LAYERS="$al" HIDDEN_DIMS="$hd" MODEL_MODULE="$m" RUN_NAME="$run_name" \
              sh "$ROOT/run.sh" >/dev/null 2>&1 &
            pids+=($!)
          done
        done
        failed=false
        for pid in "${pids[@]}"; do
          if ! wait "$pid"; then failed=true; fi
        done
        echo "--- Results for combo: vqc_layers=$v reuploading=$r attn_layers=$al hidden_dim=$hd ---"
        for tag in "${tags[@]}"; do
          run_name="${tag%%|*}"
          rest="${tag#*|}"
          ds="${rest%%|*}"
          img="${rest##*|}"
          log="$ROOT/results/logs/${run_name}.log"
          if [ ! -f "$log" ]; then
            echo "[$ds img$img] MISSING_LOG $run_name"
            continue
          fi
          metrics=$(extract_metrics "$log" || true)
          if [ -z "$metrics" ]; then
            echo "[$ds img$img] NO_METRICS $run_name"
            continue
          fi
          read -r acc auroc precision recall f1 auprc <<<"$metrics"
          printf "[%s img%s] acc=%s auroc=%s precision=%s recall=%s f1=%s auprc=%s (%s)\n" \
            "$ds" "$img" "$acc" "$auroc" "$precision" "$recall" "$f1" "$auprc" "$run_name"
        done
        if $failed; then
          echo "One or more runs failed for combo v=$v r=$r al=$al hd=$hd (parallel models)"; continue
        fi
        all_pass=true
        for tag in "${tags[@]}"; do
          run_name="${tag%%|*}"
          rest="${tag#*|}"
          ds="${rest%%|*}"
          img="${rest##*|}"
          log="$ROOT/results/logs/${run_name}.log"
          if [ ! -f "$log" ]; then all_pass=false; break; fi
          metrics=$(extract_metrics "$log") || { all_pass=false; break; }
          key="${ds}_${img}"
          if ! meets_thresholds "$key" $metrics; then
            all_pass=false; break
          fi
        done
        if $all_pass; then
          echo "Found satisfying combo: vqc_layers=$v reuploading=$r attn_layers=$al hidden_dim=$hd"
          exit 0
        fi
      done
    done
  done
done

echo "No combination met all thresholds."
exit 1
