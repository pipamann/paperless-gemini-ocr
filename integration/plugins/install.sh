#!/bin/bash
# Runs as root before Paperless starts (custom container initialisation).
set -e
pip install --no-index --no-deps /opt/gemini-ocr/paperless_gemini_ocr-*.whl
