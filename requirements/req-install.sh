#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#── COLORS & LOGGING ────────────────────────────────────────────────
GREEN="$(tput setaf 2)";   YELLOW="$(tput setaf 3)"
RED="$(tput setaf 1)";     BLUE="$(tput setaf 4)"
BOLD="$(tput bold)";       RESET="$(tput sgr0)"

log_info()    { printf "%s%s[INFO ]%s %s\n"    "${BLUE}"  "${BOLD}" "${RESET}" "$*"; }
log_warn()    { printf "%s%s[WARN ]%s %s\n"    "${YELLOW}" "${BOLD}" "${RESET}" "$*"; }
log_error()   { printf "%s%s[ERROR]%s %s\n"    "${RED}"   "${BOLD}" "${RESET}" "$*"; exit 1; }
log_success() { printf "%s%s[SUCCESS]%s %s\n"  "${GREEN}" "${BOLD}" "${RESET}" "$*"; }

#── ENSURE NOT IN BASE ENV ─────────────────────────────────────────
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    log_error "No active conda/micromamba environment detected. Please activate one first."
fi
# Also guard against CONDA_DEFAULT_ENV if available
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]] || [[ "$(basename "$CONDA_PREFIX")" == "base" ]]; then
    log_error "Refusing to modify the 'base' environment. Activate a non‑base env and retry."
fi
env_name="$(basename "$CONDA_PREFIX")"
log_info "Active environment: $env_name"

#── ENVIRONMENT DETECTION ───────────────────────────────────────────
IS_APPLE_SILICON=false
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    IS_APPLE_SILICON=true
fi

if command -v mamba &>/dev/null; then
    PKG_MANAGER="mamba"
elif command -v micromamba &>/dev/null; then
    PKG_MANAGER="micromamba"
else
    PKG_MANAGER="conda"
fi
log_info "Using package manager: $PKG_MANAGER"

#── INSTALL FUNCTIONS ───────────────────────────────────────────────
install_kilosort_cpu() {
    log_info "Installing Kilosort (CPU) → python3.10 + scripts [conversion/plotting only]"
    $PKG_MANAGER install -n "$env_name" python=3.10 -y
    pip install osqp
    pip install kilosort[gui]
    pip install -r kilo-scripts.txt
    log_success "Kilosort (CPU) setup complete"
}

install_kilosort_gpu() {
    if $IS_APPLE_SILICON; then
        log_error "Kilosort GPU not supported on Apple Silicon"
    fi
    if [[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        log_error "No GPU detected. Ensure you're on a GPU partition."
    fi
    log_info "Installing Kilosort (GPU) → python3.10 + scripts [recommended for sorting]"
    $PKG_MANAGER install -n "$env_name" python=3.10 -y
    pip install kilosort[gui]
    pip install -r kilo-scripts.txt
    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    log_success "Kilosort (GPU) setup complete"
}

install_phy() {
    if $IS_APPLE_SILICON; then
        log_error "Phy not supported on Apple Silicon"
    fi
    log_info "Installing Phy → python3.11 + phy-base.txt [viewing results only]"
    $PKG_MANAGER install -n "$env_name" python=3.11 -y
    pip install -r phy-base.txt
    log_success "Phy setup complete"
}

#── MAIN MENU ───────────────────────────────────────────────────────
echo
printf "%s%sSelect environment to set up:%s\n\n" "${BOLD}" "${BLUE}" "${RESET}"
printf "  %s1.%s Kilosort (CPU) and kilosort‑scripts %s[Recommended for conversion/plotting functions only]%s\n" \
       "${GREEN}" "${RESET}" "${BOLD}" "${RESET}"
if $IS_APPLE_SILICON; then
    printf "  %s2.%s Kilosort (GPU) %s[NOT AVAILABLE on Apple Silicon]%s\n" \
           "${YELLOW}" "${RESET}" "${RED}" "${RESET}"
    printf "  %s3.%s Phy           %s[NOT AVAILABLE on Apple Silicon]%s\n" \
           "${YELLOW}" "${RESET}" "${RED}" "${RESET}"
else
    printf "  %s2.%s Kilosort (GPU) and kilosort‑scripts %s[Recommended for sorting]%s\n" \
           "${GREEN}" "${RESET}" "${BOLD}" "${RESET}"
    printf "  %s3.%s Phy %s[Viewing results only]%s\n" \
           "${GREEN}" "${RESET}" "${BOLD}" "${RESET}"
fi
echo
read -rp "Enter choice (1/2/3): " choice
echo

case "$choice" in
    1) install_kilosort_cpu ;;
    2) install_kilosort_gpu ;;
    3) install_phy ;;
    *) log_error "Invalid choice: $choice" ;;
esac

log_success "All done! Environment '$env_name' is ready."