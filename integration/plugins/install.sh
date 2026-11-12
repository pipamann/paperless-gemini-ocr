#!/bin/bash
# Runs as root before Paperless starts (custom container initialisation).
set -e
wheel=$(ls /opt/gemini-ocr/paperless_gemini_ocr-*.whl 2>/dev/null | sort -V | tail -n 1)
if [ -z "$wheel" ]; then
    echo "gemini-ocr: no wheel in /opt/gemini-ocr, the plugin is NOT installed" >&2
    exit 1
fi
pip install --no-index --no-deps --force-reinstall "$wheel"
