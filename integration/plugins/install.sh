#!/bin/bash
# Runs as root before Paperless starts (custom container initialisation).
set -e
pip install --no-index --no-deps /custom-cont-init.d/paperless_gemini_ocr-*.whl
