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
- A PDF page counts as a scan when an image covers most of it, even if the
  scanner added its own text layer. Whatever is doubtful (no text layer, a
  large image with only a caption, content the analysis cannot see into) goes
  to the model, because calling a scan "digital" would lose its content.
- If the model fails for a page, the import still succeeds: Tesseract reads
  exactly the failed pages, and the document is tagged `ocr-review` with a
  note naming them. Nothing is silently worse.
- "Reprocess" on an existing document uses the plugin too. There a model
  failure aborts the reprocessing, so good text is never replaced by fallback
  text: the document stays as it was and Paperless logs the error.
- Pages the model reports as upside down are turned upright in the archive
  PDF. The original file is never modified.
- A page the model calls empty is not taken on its word. If the page has a
  text layer of its own, the page counts as failed and keeps that text.
  Otherwise Tesseract takes a quick look at a small rendering of the page
  (about a second per page): if it sees text, or could not
  look, the page counts as failed too, gets proper local OCR text, and the
  document is tagged. Scanner specks on a blank page are not taken for text.
  The look uses the OCR languages of the installation, so it cannot vouch
  for scripts Tesseract is not set up for. With `PAPERLESS_OCR_MODE=off`
  there is nothing to check against.
- A rate limit that says how long to wait (HTTP 429 with a retry delay) is
  waited out, up to two minutes and never past the document's time limit,
  instead of using up the attempts within seconds.

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
   `docker-compose.yml`, for example `./gemini-ocr/`.
2. Put an install script into the same folder, `./gemini-ocr/install.sh`:

   ```bash
   #!/bin/bash
   set -e
   wheel=$(ls /opt/gemini-ocr/paperless_gemini_ocr-*.whl 2>/dev/null | sort -V | tail -n 1)
   if [ -z "$wheel" ]; then
       echo "gemini-ocr: no wheel in /opt/gemini-ocr, the plugin is NOT installed" >&2
       exit 1
   fi
   pip install --no-index --no-deps --force-reinstall "$wheel"
   ```

   The script must belong to root and must not be writable by others
   (`sudo chown root:root install.sh && sudo chmod 755 install.sh`).
   Paperless skips all custom init scripts otherwise and starts without the
   plugin. The plugin only needs libraries Paperless-ngx already ships, so
   nothing is downloaded when the container starts.
3. Mount both and set the API key. Paperless runs every file it finds in
   `/custom-cont-init.d`, so only the script goes there:

   ```yaml
   services:
     webserver:
       volumes:
         - ./gemini-ocr/install.sh:/custom-cont-init.d/install-gemini-ocr.sh:ro
         - ./gemini-ocr:/opt/gemini-ocr:ro
       environment:
         PAPERLESS_GEMINI_OCR_API_KEY: ${GEMINI_API_KEY}
   ```

4. Recreate the container and look for this line in the log:

   ```
   Loaded third-party parser 'Gemini OCR' v0.1.0 by pipamann (entrypoint: 'gemini_ocr').
   ```

   If the line is missing, the plugin is not installed and every document is
   read by Tesseract as before; the container log then says why
   (`[custom-init]` lines).

Without an API key the plugin stays inactive and Paperless behaves as before.
To remove it, delete the two mounts and recreate the container.

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
| `PAPERLESS_GEMINI_OCR_ROTATE` | `upside-down` | `upside-down`: turn pages the model reports as rotated by 180° upright in the archive PDF. `all`: also 90° and 270° (the model sometimes confuses the two). `off`: never. |
| `PAPERLESS_GEMINI_OCR_FAILURE_TAG` | `ocr-review` | Tag for documents with failed pages; empty disables it. |
| `PAPERLESS_GEMINI_OCR_FAILURE_NOTE` | `true` | Add a note naming the failed pages. |
| `PAPERLESS_GEMINI_OCR_PROMPT_FILE` | – | Path to your own prompt. |
| `PAPERLESS_GEMINI_OCR_BASE_URL` | Google | API base URL, for proxies and tests. |

### The trade-off behind `LOCAL_OCR`

With `fallback`, a scan's archive PDF has no text layer: Paperless' search
works on the model's text, but searching inside the downloaded PDF does not.
In exchange Tesseract, usually the slowest step of an import, is skipped.
With `always` you keep the text layer and the old import time. Turning pages
is then left to OCRmyPDF (`PAPERLESS_OCR_ROTATE_PAGES`, on by default); the
model's finding is only used when that setting is off.

### Limits

Files far outside what a scanner produces are not sent to the model, or only
in reduced form: a rendered page has at most 5000 pixels on its longer edge,
an image with more than 120 megapixels or a TIFF with more than 2000 frames
is imported with local OCR only and tagged, and a PDF page whose content
needs unusual effort to analyse is rendered instead of being taken apart.

The Gemini API refuses a page now and then, for example a handwritten page
with ID numbers (`blockReason: OTHER`). That is final for the page: Tesseract
reads it and the document is tagged.

## Speed and cost

Measured on a small VPS with `gemini-3.8-flash`: a single scanned page takes
3 to 6 seconds, and the pages of a document are read side by side, so a
six-page scan took 11 seconds. A page used about 1,650 input tokens and 200
to 1,100 output tokens, depending on how much text is on it.

## Development

```bash
uv venv && uv pip install -e ".[test]"
.venv/bin/python -m pytest
```

The unit tests need `pdftotext` and `pdftoppm` (poppler); a few also use
`tesseract` and are skipped without it. They run against
stand-ins for the Paperless modules; `integration/` holds the tests against a
real Paperless-ngx container.

## License

MIT
