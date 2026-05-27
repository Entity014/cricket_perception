#!/usr/bin/env bash
# ===========================================================================
#  download_dataset.sh
#  Downloads InsectSet32 from Zenodo into dataset/insectset32/
#
#  InsectSet32:  ~335 audio files, 32 Orthoptera/Cicadidae species
#  Record ID:    7072196  (NOTE: 7074465 is a stale redirect)
#  API Base:     https://zenodo.org/api/records/7072196/files/<name>/content
#  License:      CC BY 4.0
#
#  Usage:
#    chmod +x scripts/download_dataset.sh
#    ./scripts/download_dataset.sh
#
#  Files downloaded:
#    Orthoptera.zip  (~130 MB) — crickets & katydids  ← main for this project
#    Cicadidae.zip   (~138 MB) — cicadas              (optional)
#    Orthoptera.csv             — metadata
#    Cicadidae.csv              — metadata
#    README.txt
# ===========================================================================

set -euo pipefail

RECORD_ID="7072196"
API_BASE="https://zenodo.org/api/records/${RECORD_ID}/files"
DEST_DIR="dataset/insectset32"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; exit 1; }

command -v curl  >/dev/null 2>&1 || err "curl is required: sudo apt install curl"
command -v unzip >/dev/null 2>&1 || err "unzip is required: sudo apt install unzip"

mkdir -p "${DEST_DIR}"

# ── Download helper ───────────────────────────────────────────────────────────
download_file() {
    local name="$1"
    local dest="${DEST_DIR}/${name}"
    local url="${API_BASE}/${name}/content"

    if [[ -f "${dest}" ]]; then
        log "Already exists: ${name} — skipping."
        return
    fi
    log "Downloading ${name}…"
    curl -L --progress-bar --retry 3 --retry-delay 5 \
         -o "${dest}" "${url}"
    log "Done: $(du -sh "${dest}" | cut -f1) — ${name}"
}

# ── Download files ────────────────────────────────────────────────────────────
log "=== InsectSet32 Dataset (Zenodo record ${RECORD_ID}) ==="

download_file "README.txt"
download_file "Orthoptera.csv"
download_file "Orthoptera.zip"    # ← crickets & katydids (main)

# Uncomment to also download cicadas:
# download_file "Cicadidae.csv"
# download_file "Cicadidae.zip"

# ── Extract ──────────────────────────────────────────────────────────────────
log "Extracting Orthoptera.zip…"
unzip -q -o "${DEST_DIR}/Orthoptera.zip" -d "${DEST_DIR}/Orthoptera"
log "Extraction complete."

# ── Summary ──────────────────────────────────────────────────────────────────
WAV_COUNT=$(find "${DEST_DIR}" \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) | wc -l)
SPECIES_COUNT=$(find "${DEST_DIR}/Orthoptera" -mindepth 1 -maxdepth 1 -type d | wc -l)

log "────────────────────────────────────────"
log "✅  Dataset ready: ${DEST_DIR}/Orthoptera"
log "    Species folders : ${SPECIES_COUNT}"
log "    Audio files     : ${WAV_COUNT}"
log "────────────────────────────────────────"
log "Next step:"
log "    source .venv/bin/activate"
log "    jupyter notebook notebooks/01_explore_dataset.ipynb"
