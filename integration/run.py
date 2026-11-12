"""Integration tests against a running Paperless-ngx with the plugin installed.

Runs inside the webserver container (it has httpx, pikepdf and Pillow):

    docker compose exec -T webserver python3 /it/run.py functional
    docker compose exec -T webserver python3 /it/run.py load --count 24 --label fallback

All documents are generated here; nothing real is involved. The Gemini API is
the stand-in from fake_gemini.py.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
import pikepdf
from pikepdf import Dictionary
from pikepdf import Name
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FAKE = "http://fake-gemini:8080"
CONSUME = Path("/usr/src/paperless/consume")
RESULTS = Path("/results")
LOG = Path("/usr/src/paperless/data/log/paperless.log")
api = httpx.Client(
    base_url="http://localhost:8000",
    auth=("admin", "integration-test"),
    headers={"Accept": "application/json; version=9"},
    timeout=120,
)


# ---------------------------------------------------------------------------
# Generated documents
# ---------------------------------------------------------------------------


def uid() -> str:
    return uuid.uuid4().hex[:10]


def page_image(lines: list[str], dpi: int = 150, noise: bool = False) -> Image.Image:
    width, height = round(8.27 * dpi), round(11.69 * dpi)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=round(dpi * 0.19))
    for index, line in enumerate(lines):
        draw.text((dpi * 0.8, dpi * (1 + index * 0.4)), line, fill="black", font=font)
    if noise:
        grain = Image.effect_noise((width, height), 14).convert("RGB")
        image = Image.blend(image, grain, 0.07)
    return image


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def tiff(images: list[Image.Image]) -> bytes:
    """A multi-page TIFF, as a fax server or some scanners write it."""
    buffer = io.BytesIO()
    frames = [image.convert("L") for image in images]
    frames[0].save(
        buffer,
        "TIFF",
        save_all=True,
        append_images=frames[1:],
        compression="tiff_deflate",
        dpi=(150, 150),
    )
    return buffer.getvalue()


def scan_pdf(images: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:], resolution=150)
    return buffer.getvalue()


def _text_stream(lines: list[str], hidden: bool = False) -> bytes:
    out = ["BT", "3 Tr" if hidden else "0 Tr", "/F1 11 Tf", "50 780 Td", "14 TL"]
    for line in lines:
        safe = line.replace("\\", "").replace("(", "").replace(")", "")
        out.append(f"({safe}) Tj T*")
    out.append("ET")
    return "\n".join(out).encode()


def _add_text_page(pdf: pikepdf.Pdf, lines: list[str], image: Image.Image | None = None) -> None:
    pdf.add_blank_page(page_size=(595, 842))
    page = pdf.pages[-1]
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica),
    )
    resources = Dictionary(Font=Dictionary(F1=font))
    content = _text_stream(lines, hidden=image is not None)
    if image is not None:
        data = jpeg(image)
        resources["/XObject"] = Dictionary(
            Im0=pikepdf.Stream(
                pdf,
                data,
                Type=Name.XObject,
                Subtype=Name.Image,
                Width=image.width,
                Height=image.height,
                ColorSpace=Name.DeviceRGB,
                BitsPerComponent=8,
                Filter=Name.DCTDecode,
            ),
        )
        content = b"q 595 0 0 842 0 0 cm /Im0 Do Q\n" + content
    page.obj["/Resources"] = resources
    page.obj["/Contents"] = pikepdf.Stream(pdf, content)


def text_pdf(pages: list[list[str]]) -> bytes:
    pdf = pikepdf.Pdf.new()
    for lines in pages:
        _add_text_page(pdf, lines)
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def scanner_pdf(lines: list[str]) -> bytes:
    """A scan with the scanner's own invisible text layer, as ScanSnap writes it."""
    pdf = pikepdf.Pdf.new()
    _add_text_page(pdf, lines, image=page_image(lines))
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def mixed_pdf(text_lines: list[str], scan_lines: list[str]) -> bytes:
    pdf = pikepdf.Pdf.new()
    _add_text_page(pdf, text_lines)
    with pikepdf.open(io.BytesIO(scan_pdf([page_image(scan_lines)]))) as scan:
        pdf.pages.extend(scan.pages)
        buffer = io.BytesIO()
        pdf.save(buffer)
    return buffer.getvalue()


def letter(marker: str) -> list[str]:
    return [
        "Musterfirma GmbH",
        f"Rechnung Nummer {marker}",
        "Datum 03.11.2026",
        "Beratung drei Stunden 360,00 EUR",
        "Fahrtkosten 48,50 EUR",
        "Gesamtbetrag 486,12 EUR",
    ]


# ---------------------------------------------------------------------------
# Paperless and stand-in helpers
# ---------------------------------------------------------------------------


def control(**params) -> None:
    httpx.get(f"{FAKE}/control", params=params, timeout=30).raise_for_status()


def stats() -> dict:
    return httpx.get(f"{FAKE}/stats", timeout=30).json()


def upload(name: str, data: bytes) -> None:
    response = api.post("/api/documents/post_document/", files={"document": (name, data)})
    response.raise_for_status()


def find(title: str) -> list[dict]:
    response = api.get("/api/documents/", params={"title__icontains": title, "page_size": 100})
    response.raise_for_status()
    return response.json()["results"]


def wait_for(title: str, timeout: float = 420) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = find(title)
        if found:
            return found[0]
        time.sleep(1.5)
    raise TimeoutError(f"document {title} did not appear within {timeout:.0f} s")


def consume(case: str, suffix: str, data: bytes) -> dict:
    title = f"{case}-{uid()}"
    upload(f"{title}.{suffix}", data)
    return wait_for(title)


def document(doc_id: int) -> dict:
    response = api.get(f"/api/documents/{doc_id}/")
    response.raise_for_status()
    return response.json()


def review_tag_id() -> int | None:
    found = api.get("/api/tags/", params={"name__iexact": "ocr-review"}).json()["results"]
    return found[0]["id"] if found else None


def notes(doc_id: int) -> list[str]:
    response = api.get(f"/api/documents/{doc_id}/notes/")
    response.raise_for_status()
    return [n["note"] for n in response.json()]


def archive_rotation(doc_id: int) -> list[int]:
    response = api.get(f"/api/documents/{doc_id}/download/")
    response.raise_for_status()
    with pikepdf.open(io.BytesIO(response.content)) as pdf:
        return [int(page.obj.get("/Rotate", 0)) for page in pdf.pages]


def reprocess(doc_id: int) -> None:
    response = api.post(
        "/api/documents/bulk_edit/",
        json={"documents": [doc_id], "method": "reprocess", "parameters": {}},
    )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

CHECKS: list[dict] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    CHECKS.append({"check": name, "ok": bool(condition), "detail": str(detail)[:400]})
    print(("PASS  " if condition else "FAIL  ") + name + (f"  [{detail}]" if not condition else ""))
    sys.stdout.flush()


def functional() -> None:
    # --- born-digital PDF: embedded text, nothing leaves the server
    control(mode="ok")
    marker = uid()
    doc = document(consume("digital", "pdf", text_pdf([letter(marker), letter(marker)]))["id"])
    check("born-digital: embedded text kept", marker in doc["content"], doc["content"][:120])
    check("born-digital: nothing sent to the model", stats()["requests"] == 0, stats())

    # --- JPG scan
    control(mode="ok")
    doc = document(consume("scan-jpg", "jpg", jpeg(page_image(letter(uid()))))["id"])
    jpg_doc_id = doc["id"]
    seen = stats()
    check("jpg scan: model text is the content", doc["content"].startswith("FAKE-OCR request"), doc["content"][:120])
    check("jpg scan: one request, as image/jpeg", seen["requests"] == 1 and seen["mimes"] == ["image/jpeg"], seen)
    check("jpg scan: API key sent as header", seen["key_seen"], seen)
    check("jpg scan: archive file exists", bool(doc.get("archived_file_name")), doc.get("archived_file_name"))
    check("jpg scan: not flagged", review_tag_id() not in doc["tags"], doc["tags"])
    check("jpg scan: archive not rotated", archive_rotation(doc["id"]) == [0])

    # --- upside-down JPG: archive turned, original untouched
    control(mode="ok", rotation=180)
    original = jpeg(page_image(letter(uid())).rotate(180))
    doc = document(consume("upside-down", "jpg", original)["id"])
    check("upside-down: archive page turned by 180", archive_rotation(doc["id"]) == [180], archive_rotation(doc["id"]))
    stored = api.get(f"/api/documents/{doc['id']}/download/", params={"original": "true"}).content
    check("upside-down: original file unchanged", stored == original, f"{len(stored)} vs {len(original)} bytes")

    # --- 3-page scan PDF: one request per page, in parallel
    control(mode="ok", delay=2)
    marker = uid()
    data = scan_pdf([page_image(letter(f"{marker}-{n}")) for n in (1, 2, 3)])
    doc = document(consume("scan-pdf", "pdf", data)["id"])
    seen = stats()
    check("scan pdf: three pages read", doc["content"].count("FAKE-OCR request") == 3, doc["content"])
    check("scan pdf: pages sent as single-page PDFs", seen["mimes"] == ["application/pdf"] * 3, seen)
    check("scan pdf: pages read in parallel", seen["max_parallel"] >= 2, seen)
    check("scan pdf: page count", doc.get("page_count") == 3, doc.get("page_count"))

    # --- scan with the scanner's own text layer still goes to the model
    control(mode="ok")
    marker = uid()
    doc = document(consume("scanner-layer", "pdf", scanner_pdf(letter(marker)))["id"])
    check("scanner text layer: model text wins", "FAKE-OCR request" in doc["content"] and marker not in doc["content"], doc["content"][:160])
    check("scanner text layer: one request", stats()["requests"] == 1, stats())

    # --- mixed PDF: page 1 embedded, page 2 from the model
    control(mode="ok")
    marker = uid()
    doc = document(consume("mixed", "pdf", mixed_pdf(letter(marker), letter(uid())))["id"])
    check("mixed: embedded page kept", marker in doc["content"], doc["content"][:160])
    check("mixed: scanned page from the model", "FAKE-OCR request" in doc["content"], doc["content"][-160:])
    check("mixed: page order", doc["content"].find(marker) < doc["content"].find("FAKE-OCR"), "")
    check("mixed: only the scanned page was sent", stats()["requests"] == 1, stats())

    # --- model down: import succeeds with local OCR, document flagged
    control(mode="fail")
    # Words as marks: Tesseract misreads about one random number in ten.
    first, second = "Elster", "Fuchs"
    data = scan_pdf([page_image(letter(mark)) for mark in (first, second)])
    started = time.monotonic()
    doc = document(consume("model-down", "pdf", data)["id"])
    elapsed = time.monotonic() - started
    tag = review_tag_id()
    doc_notes = notes(doc["id"])
    check("model down: document imported anyway", True, f"{elapsed:.0f} s")
    check("model down: content is local OCR", "FAKE-OCR" not in doc["content"] and "Musterfirma" in doc["content"], doc["content"][:200])
    check("model down: both pages in page order", -1 < doc["content"].find(first) < doc["content"].find(second), doc["content"][:300])
    check("model down: tagged ocr-review", tag is not None and tag in doc["tags"], f"tag={tag} tags={doc['tags']}")
    check("model down: note names the pages", any("page(s) 1, 2 of 2" in n for n in doc_notes), doc_notes)
    check("model down: three attempts per page", stats()["requests"] == 6, stats())
    check("model down: found by tag filter", any(d["id"] == doc["id"] for d in api.get("/api/documents/", params={"tags__id__all": tag}).json()["results"]) if tag else False, "")

    # --- model down, mixed PDF: Paperless sees text and would skip OCR; the
    #     scanned page must get OCR text all the same
    control(mode="fail")
    digital, scanned = uid(), uid()
    doc = document(consume("mixed-down", "pdf", mixed_pdf(letter(digital), letter(scanned)))["id"])
    check("model down, mixed: embedded page kept", digital in doc["content"], doc["content"][:160])
    check("model down, mixed: scanned page read by local OCR", "Gesamtbetrag" in doc["content"].split(digital, 1)[-1].split("Gesamtbetrag", 1)[-1] or doc["content"].count("Gesamtbetrag") == 2, doc["content"][-300:])
    check("model down, mixed: flagged", review_tag_id() in doc["tags"], doc["tags"])

    # --- model down, scan with scanner text layer (Paperless produces no archive)
    control(mode="fail")
    marker = uid()
    doc = document(consume("layer-down", "pdf", scanner_pdf(letter(marker)))["id"])
    check("model down, scanner layer: text present", "Musterfirma" in doc["content"], doc["content"][:200])
    check("model down, scanner layer: flagged", review_tag_id() in doc["tags"], doc["tags"])

    # --- model down, JPG: archive is still produced and valid
    control(mode="fail")
    doc = document(consume("jpg-down", "jpg", jpeg(page_image(letter(uid()))))["id"])
    check("model down, jpg: local OCR text", "Musterfirma" in doc["content"], doc["content"][:120])
    check("model down, jpg: archive readable", archive_rotation(doc["id"]) == [0], "")

    # --- one failed attempt, then fine: no flag
    control(mode="flaky", fail_first=1)
    doc = document(consume("flaky", "jpg", jpeg(page_image(letter(uid()))))["id"])
    check("flaky: retried and read", doc["content"].startswith("FAKE-OCR request 2"), doc["content"][:80])
    check("flaky: not flagged", review_tag_id() not in doc["tags"], doc["tags"])

    # --- empty answers count as failures
    control(mode="empty")
    doc = document(consume("empty", "jpg", jpeg(page_image(letter(uid()))))["id"])
    check("empty answers: local OCR and flag", "Musterfirma" in doc["content"] and review_tag_id() in doc["tags"], doc["content"][:120])

    # --- multi-page TIFF: one request per frame
    control(mode="ok")
    doc = document(consume("tiff", "tif", tiff([page_image(letter(uid())) for _ in range(3)]))["id"])
    seen = stats()
    check("tiff: three frames read", doc["content"].count("FAKE-OCR request") == 3, doc["content"])
    check("tiff: frames sent as images", seen["mimes"] == ["image/jpeg"] * 3, seen)

    # --- multi-page TIFF, model down: every frame read locally, once, in order
    control(mode="fail")
    # Words, not random numbers: Tesseract misreads about one number in ten
    # (a blank inside, O for 0), whatever the file type.
    marks = ["Adler", "Biber", "Dachs"]
    doc = document(consume("tiff-down", "tif", tiff([page_image(letter(m)) for m in marks]))["id"])
    content = doc["content"]
    places = [content.find(m) for m in marks]
    check("model down, tiff: all frames read by local OCR", all(p >= 0 for p in places), content[:300])
    check("model down, tiff: frames in order", places == sorted(places), places)
    check("model down, tiff: no frame twice", all(content.count(m) == 1 for m in marks), [content.count(m) for m in marks])
    check("model down, tiff: flagged", review_tag_id() in doc["tags"], doc["tags"])

    # --- upside-down scan PDF: the archive is turned there too
    control(mode="ok", rotation=180)
    doc = document(consume("upside-down-pdf", "pdf", scan_pdf([page_image(letter(uid())) for _ in range(2)]))["id"])
    check("upside-down pdf: archive pages turned by 180", archive_rotation(doc["id"]) == [180, 180], archive_rotation(doc["id"]))

    # --- a page the model calls empty: believed for a plain image ...
    control(mode="blank")
    doc = document(consume("blank", "jpg", jpeg(page_image([])))["id"])
    check("blank page: imported without text and without flag", doc["content"].strip() == "" and review_tag_id() not in doc["tags"], doc["content"][:80])

    # ... but not for a scan whose own text layer has text
    control(mode="blank")
    marker = uid()
    doc = document(consume("blank-layer", "pdf", scanner_pdf(letter(marker)))["id"])
    check("blank answer, scanner layer: text kept", "Musterfirma" in doc["content"], doc["content"][:200])
    check("blank answer, scanner layer: flagged", review_tag_id() in doc["tags"], doc["tags"])

    # ... and not for a plain scan on which local OCR does find text
    control(mode="blank")
    doc = document(consume("blank-claim", "jpg", jpeg(page_image(letter(uid()))))["id"])
    check("blank answer, text on the page: local OCR text", "Musterfirma" in doc["content"], doc["content"][:200])
    check("blank answer, text on the page: flagged", review_tag_id() in doc["tags"], doc["tags"])

    # In a PDF only the page with text counts as failed, the empty one stays empty.
    control(mode="blank")
    doc = document(consume("blank-claim-pdf", "pdf", scan_pdf([page_image([]), page_image(letter(uid()))]))["id"])
    doc_notes = notes(doc["id"])
    check("blank answer, pdf: text of the page with text", "Musterfirma" in doc["content"], doc["content"][:200])
    check("blank answer, pdf: note names that page only", any("page(s) 2 of 2" in n for n in doc_notes), doc_notes)

    # --- a rate limit that says how long to wait is waited out
    control(mode="ratelimit", fail_first=1)
    doc = document(consume("rate-limit", "jpg", jpeg(page_image(letter(uid()))))["id"])
    seen = stats()
    check("rate limit: read on the second attempt", doc["content"].startswith("FAKE-OCR request 2"), doc["content"][:80])
    check("rate limit: waited as long as asked", len(seen["times"]) == 2 and seen["times"][1] - seen["times"][0] >= 6, seen["times"])
    check("rate limit: not flagged", review_tag_id() not in doc["tags"], doc["tags"])

    # --- error answers of another shape are retried like any other failure
    control(mode="garbage")
    doc = document(consume("garbage", "jpg", jpeg(page_image(letter(uid()))))["id"])
    check("odd error body: three attempts", stats()["requests"] == 3, stats())
    check("odd error body: local OCR and flag", "Musterfirma" in doc["content"] and review_tag_id() in doc["tags"], doc["content"][:120])

    # --- reprocess: a failure leaves the content alone, a success replaces it
    before = document(jpg_doc_id)["content"]
    control(mode="fail")
    reprocess(jpg_doc_id)
    time.sleep(45)
    check("reprocess with model down: content unchanged", document(jpg_doc_id)["content"] == before, document(jpg_doc_id)["content"][:80])
    check("reprocess with model down: model was tried", stats()["requests"] == 3, stats())
    control(mode="ok", label="second-reading")
    reprocess(jpg_doc_id)
    deadline = time.monotonic() + 120
    after = before
    while time.monotonic() < deadline and after == before:
        time.sleep(2)
        after = document(jpg_doc_id)["content"]
    check("reprocess with model up: content replaced", "second-reading" in after, after[:80])

    # --- new version of an existing document goes through the plugin too
    control(mode="ok")
    marker = uid()
    response = api.post(
        f"/api/documents/{jpg_doc_id}/update_version/",
        files={"document": (f"version-{marker}.jpg", jpeg(page_image(letter(marker))))},
    )
    check("new version: accepted", response.status_code == 200, response.text[:200])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and stats()["requests"] == 0:
        time.sleep(2)
    time.sleep(5)
    check("new version: read by the model", stats()["requests"] == 1, stats())
    check("new version: document shows model text", document(jpg_doc_id)["content"].startswith("FAKE-OCR request 1"), document(jpg_doc_id)["content"][:80])

    # --- log
    log = LOG.read_text(errors="replace") if LOG.is_file() else ""
    check("log: plugin discovered", "Loaded third-party parser 'Gemini OCR'" in log, "")
    check("log: summary line per document", "Gemini OCR: 3 page(s), 3 read by" in log, "")
    check("log: no API key", "not-a-real-key" not in log, "")

    failed = [c for c in CHECKS if not c["ok"]]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "functional.json").write_text(json.dumps(CHECKS, indent=1))
    print(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# Load test
# ---------------------------------------------------------------------------


def load(count: int, label: str, delay: float, dpi: int) -> None:
    """Drop ``count`` scanner-sized JPGs at once and time until each is a document."""
    control(mode="ok", delay=delay)
    prefix = f"load-{label}-{uid()}"
    print(f"generating {count} pages at {dpi} dpi ...")
    sys.stdout.flush()
    files = []
    for index in range(count):
        lines = letter(f"{prefix}-{index}") + [f"Zeile {n} mit etwas Fliesstext zum Lesen" for n in range(20)]
        files.append((f"{prefix}-{index:03d}.jpg", jpeg(page_image(lines, dpi=dpi, noise=True))))
    size = statistics.mean(len(data) for _, data in files) / 1e6

    started = time.monotonic()
    for name, data in files:
        part = CONSUME / (name + ".part")
        part.write_bytes(data)
        part.rename(CONSUME / name)
    seen: dict[int, float] = {}
    deadline = started + 3600
    while len(seen) < count and time.monotonic() < deadline:
        time.sleep(2)
        for doc in find(prefix):
            seen.setdefault(doc["id"], time.monotonic() - started)
    times = sorted(seen.values())
    sample = document(next(iter(seen)))["content"][:60] if seen else ""
    result = {
        "label": label,
        "count": count,
        "arrived": len(times),
        "dpi": dpi,
        "mean_file_mb": round(size, 2),
        "model_delay_s": delay,
        "first_s": round(times[0], 1) if times else None,
        "median_s": round(statistics.median(times), 1) if times else None,
        "p90_s": round(times[int(len(times) * 0.9) - 1], 1) if times else None,
        "last_s": round(times[-1], 1) if times else None,
        "model_requests": stats()["requests"],
        "sample_content": sample,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"load-{label}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("functional")
    load_parser = sub.add_parser("load")
    load_parser.add_argument("--count", type=int, default=24)
    load_parser.add_argument("--label", default="run")
    load_parser.add_argument("--delay", type=float, default=8.0)
    load_parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.command == "functional":
        functional()
    else:
        load(args.count, args.label, args.delay, args.dpi)
