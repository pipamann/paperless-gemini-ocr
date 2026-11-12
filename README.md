# paperless-gemini-ocr

A parser plugin for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) 3.x
that lets Google Gemini read your scans while they are being imported.

Status: early. Written for and tested with Paperless-ngx 3.4.2.

## What it does

- Scanned pages are sent to Gemini during consumption. The document appears in
  Paperless with the model's text already in place. There is no second
  service, no queue, no trigger tag and no later overwrite of the content.
- Pages are read in parallel, one request per page, with a time limit and
  retries.
- Pages with real embedded text (born-digital PDFs) keep that text and are
  never sent anywhere. The decision is made per page, so a contract with a
  scanned signature page gets both.
- A PDF counts as a scan when an image covers most of the page, even if the
  scanner added its own text layer.
- If the model fails for a page, the import still succeeds: that page gets
  the text of the local OCR (Tesseract), and the document is tagged
  `ocr-review` with a note naming the pages. Nothing is silently worse.
- "Reprocess" on an existing document uses the plugin too. There a model
  failure aborts the task, so good text is never replaced by fallback text.
- Pages the model reports as rotated are turned upright in the archive PDF.
  The original file is never modified.

Archive PDF, thumbnail, page count and metadata come from the built-in
Tesseract parser, which this plugin extends.

## Privacy

Scanned pages are uploaded to the Google Gemini API. Do not install this if
that is not acceptable for your documents. The plugin declares
`uses_remote_service`, so Paperless' own switch applies: with
`PAPERLESS_REMOTE_OCR_MODE=workflow_only`, only documents that a workflow
marks for remote OCR are sent; everything else is processed locally.

## Installation (Docker)

1. Download the wheel from the releases page into a folder next to your
   `docker-compose.yml`, for example `./plugins/`.
2. Add an install script, `./plugins/install.sh`:

   ```bash
   #!/bin/bash
   pip install --no-index --no-deps /custom-cont-init.d/paperless_gemini_ocr-*.whl
   ```

   The plugin only needs libraries Paperless-ngx already ships, so nothing is
   downloaded when the container starts.
3. Mount the folder and set the API key:

   ```yaml
   services:
     webserver:
       volumes:
         - ./plugins:/custom-cont-init.d:ro
       environment:
         PAPERLESS_GEMINI_OCR_API_KEY: ${GEMINI_API_KEY}
   ```

4. Recreate the container and look for this line in the log:

   ```
   Loaded third-party parser 'Gemini OCR' v0.1.0 by pipamann (entrypoint: 'gemini_ocr').
   ```

Without an API key the plugin stays inactive and Paperless behaves as before.
To remove it, delete the mount and recreate the container.

## Configuration

All variables are optional except the API key.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERLESS_GEMINI_OCR_API_KEY` | – | Gemini API key. `..._API_KEY_FILE` reads it from a file. |
| `PAPERLESS_GEMINI_OCR_MODEL` | `gemini-3.8-flash` | Model id. |
| `PAPERLESS_GEMINI_OCR_THINKING` | `low` | `low`/`medium`/`high` (thinking level), a number (thinking budget) or `default`. |
| `PAPERLESS_GEMINI_OCR_MEDIA_RESOLUTION` | `high` | `low`, `medium`, `high` or `default`. |
| `PAPERLESS_GEMINI_OCR_PAGE_INPUT` | `pdf` | Send PDF pages as single-page PDFs (`pdf`) or rendered JPEGs (`image`). |
| `PAPERLESS_GEMINI_OCR_RENDER_DPI` | `200` | Resolution for rendered pages. |
| `PAPERLESS_GEMINI_OCR_CONCURRENCY` | `8` | Pages of one document read at the same time. |
| `PAPERLESS_GEMINI_OCR_REQUEST_TIMEOUT` | `90` | Seconds per request. |
| `PAPERLESS_GEMINI_OCR_MAX_ATTEMPTS` | `3` | Attempts per page. |
| `PAPERLESS_GEMINI_OCR_DOCUMENT_TIMEOUT` | `600` | Seconds for all pages of a document. |
| `PAPERLESS_GEMINI_OCR_DIGITAL_PAGES` | `embedded` | `embedded` keeps embedded text, `gemini` sends every page. |
| `PAPERLESS_GEMINI_OCR_SCAN_IMAGE_COVERAGE` | `0.7` | Share of a page covered by images from which it counts as a scan. |
| `PAPERLESS_GEMINI_OCR_LOCAL_OCR` | `fallback` | `fallback`: Tesseract only runs for pages the model failed on. `always`: Tesseract runs for every scan and the archive PDF keeps a text layer. |
| `PAPERLESS_GEMINI_OCR_ROTATE` | `true` | Turn rotated pages upright in the archive PDF. |
| `PAPERLESS_GEMINI_OCR_FAILURE_TAG` | `ocr-review` | Tag for documents with failed pages; empty disables it. |
| `PAPERLESS_GEMINI_OCR_FAILURE_NOTE` | `true` | Add a note naming the failed pages. |
| `PAPERLESS_GEMINI_OCR_PROMPT_FILE` | – | Path to your own prompt. |
| `PAPERLESS_GEMINI_OCR_BASE_URL` | Google | API base URL, for proxies and tests. |

### The trade-off behind `LOCAL_OCR`

With `fallback`, a scan's archive PDF has no text layer: Paperless' search
works on the model's text, but searching inside the downloaded PDF does not.
In exchange Tesseract, usually the slowest step of an import, is skipped.
With `always` you keep the text layer and the old import time.

## Development

```bash
uv venv && uv pip install -e ".[test]"
.venv/bin/python -m pytest
```

The unit tests need `pdftotext` and `pdftoppm` (poppler). They run against
stand-ins for the Paperless modules; `integration/` holds the tests against a
real Paperless-ngx container.

## License

MIT
