"""Pinned download of the fresh Puffer TEST dates (run ONLY after the pre-registration freeze).

Port of ``positive_controls/cohort_test_acquire.download_object`` (L249-321) and ``check_metadata`` (L213-232),
without the receipt/authority layers. For each pre-registered date (``puffer.test_dates()``):

1. GET the object metadata from the GCS JSON API (``metadata_url``) and validate it (:func:`check_metadata`);
2. GET the media pinned to that object generation (``alt=media&generation=G&ifGenerationMatch=G
   &ifMetagenerationMatch=MG``, ``Accept-Encoding`` = the storage encoding so GCS does not transcode), check the
   response headers against the pinned metadata (:func:`check_media_headers`), and stream exactly ``size`` stored
   (gzip) bytes to ``<target>.partial`` while hashing, followed by a one-byte EOF probe;
3. accept only if the byte count equals the metadata size, the stream ends there and the MD5 equals ``md5Hash``
   (a mismatch is retried once); then rename to ``data/puffer_raw/<date>/video_size.csv.gz``, write
   ``object_metadata.json`` next to it and check the content by preparing the date at once
   (``puffer.prepare_date``; a content failure moves the object to ``*.invalid`` and makes the date corrupt).

HTTP redirects are refused (:func:`default_opener`), so a response can only come from the pinned GCS endpoint.

Pre-registered handling (:data:`DATE_RULES`): the number of TEST dates (16, or 24 with the extension) is the freeze
field ``test_dates``; a missing or corrupt date (:data:`CORRUPT_CLAUSE`) is replaced by the next calendar day that
is neither a TEST date (including the 24-date extension) nor an earlier replacement, and after
``MAX_REPLACEMENTS`` = 7 failed candidates the fetch stops (a logged deviation). Replacements are recorded in
``data/puffer_raw/manifest.json``, which :func:`slicemech.data.puffer.split_dates` reads. The disk guard
(``io.require_free_gb(3)``) runs before every object.

Freeze gate (:func:`_require_freeze`; round-2 review: a renamed copy of the draft used to pass). Downloads, and every
later TEST read such as ``calibrate.data_report(test_freeze=...)``, require the frozen pre-registration file
(``--freeze``, under ``prereg/``):

* not a draft: no ``DRAFT`` in its name or first line, no unresolved ``⟨DEV`` placeholder, and bytes different from
  ``prereg/PREREG_DRAFT.md``;
* exactly one fenced ```` ```freeze ```` YAML block (write it with :func:`freeze_block`) with
  ``analysis_code_sha256`` (:func:`analysis_code_sha256`: every file of ``code/slicemech`` and ``code/experiments``
  except dot-files and dot-directories, ``__pycache__`` and ``*.pyc``), ``config_sha256`` (:func:`config_sha256`:
  ``spec/params.yaml``, ``spec/controllers.yaml`` and the calibrations in ``data/calibration``), ``checkpoints``
  (path relative to the project root -> SHA-256, at least one), a non-empty ``date_rule`` and ``test_dates``
  (16 or 24, consistent with the rule's date range);
* every listed checkpoint exists with that SHA-256; before the download the code and config hashes must equal the
  current tree (``check_code=True``);
* the freeze file's own SHA-256 is stored in the fetch manifest on the first download and re-checked by every later
  call, so the freeze cannot change after the TEST data exists. Once recorded, every later TEST read re-checks the
  code and config hashes against the current tree (round-3 review: they used to be skipped, so code or settings could
  change after looking at TEST results without a trace). A difference is accepted only when ``prereg/DEVIATIONS.md``
  names the current hash (a logged post-freeze fix, :func:`logged_deviation`); the returned record carries the
  current hashes so drivers can store them next to their outputs.
* :func:`require_test_read_gate` serves callers without a freeze argument (the D2 per-date count rule
  ``calibrate.counts_for_date('test', ...)``): it uses the freeze recorded in the fetch manifest and raises
  PermissionError while none is recorded.

* after the download, every TEST read also checks that the fetch manifest holds the frozen number of dates.

CLI: ``python -m slicemech.data.puffer_fetch --freeze ../prereg/PREREG.md`` (never run by tests; the date count
comes from the freeze, ``--extended`` is only a cross-check); ``--freeze-block --test-dates 16|24`` prints the block
for the current tree (with ``--checkpoint PATH`` repeated; ``--date-rule`` defaults to ``DATE_RULES[test_dates]``).
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re

import yaml
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .. import io as sio
from . import puffer

BUCKET = "puffer-data-release"
META_FIELDS = "bucket,name,size,generation,metageneration,md5Hash,crc32c,contentType,contentEncoding"
OBJECT_CAP = 500_000_000            # bytes; a video_size object is ~38 MB gzip-stored
PREREG_DIR = sio.ROOT / "prereg"
#: Directories whose files make up the analysis code hash, and files/globs of the frozen configuration.
CODE_DIRS = (sio.ROOT / "code" / "slicemech", sio.ROOT / "code" / "experiments")
CONFIG_FILES = (sio.ROOT / "spec" / "params.yaml", sio.ROOT / "spec" / "controllers.yaml")
CONFIG_GLOBS = ((sio.DATA_DIR / "calibration", "*.json"),)
FREEZE_FIELDS = ("analysis_code_sha256", "config_sha256", "checkpoints", "date_rule", "test_dates")
#: The two pre-registered TEST date counts (16; 24 with the power-triggered extension). The count is a freeze field
#: (pre-freeze audit F5): the download derives the date list from it, and every later TEST read re-checks it.
TEST_DATE_COUNTS = (16, 24)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FREEZE_BLOCK = re.compile(r"^```freeze[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
USER_AGENT = "slicemech-puffer-fetch/1"
MAX_REPLACEMENTS = 7
MD5_ATTEMPTS = 2                    # a size/MD5 mismatch is retried once before the date counts as corrupt (F3)
BLOCK = 1 << 20                     # bytes per streamed read

#: Pre-registered replacement rule (PREREG §2; pre-freeze audit E5-3/F4): the code is :func:`replacement_date`
#: (taken = every TEST date including the extension, plus the earlier replacements) and ``MAX_REPLACEMENTS``.
REPLACEMENT_CLAUSE = ("a missing or corrupt date is replaced by the next calendar day that is neither a TEST date "
                      "(including the 24-date extension) nor an earlier replacement; after 7 failed candidates the "
                      "fetch stops and the failure is logged as a deviation")
#: Pre-registered meaning of 'missing' and 'corrupt' (pre-freeze audit F3; code: :func:`fetch_date`).
CORRUPT_CLAUSE = ("a date is missing when its object metadata returns HTTP 404, and corrupt when its object fails "
                  "the pinned size/MD5 check on two attempts, cannot be decompressed or parsed, fails a validation "
                  "of puffer.parse_video_size, or has none of the seven channels with all four formats")
#: The freeze-block ``date_rule`` for each pre-registered date count (EXPERIMENTS.md §4.3 quotes them).
DATE_RULES = {
    16: f"Sundays and Mondays 2024-09-01..2024-10-21 (16 dates); {REPLACEMENT_CLAUSE}; {CORRUPT_CLAUSE}",
    24: f"Sundays and Mondays 2024-09-01..2024-11-18 (24 dates); {REPLACEMENT_CLAUSE}; {CORRUPT_CLAUSE}",
}
_RULE_END = {16: "2024-10-21", 24: "2024-11-18"}     # last TEST date named by each rule


class CorruptObject(ValueError):
    """The date counts as corrupt (:data:`CORRUPT_CLAUSE`): the downloaded bytes do not match the pinned size/MD5
    on two attempts, or the object fails the content check (``puffer.InvalidVideoSize``)."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"HTTP redirect to {newurl} refused (pinned GCS download)")


def default_opener():
    """``opener(request, timeout=...)`` that refuses HTTP redirects."""
    return build_opener(_NoRedirect()).open


def manifest_path() -> pathlib.Path:
    """Fetch manifest (objects, replacements, failures) read by :func:`slicemech.data.puffer.effective_test_dates`."""
    return puffer.TEST_RAW_DIR / "manifest.json"


def metadata_url(day: str) -> str:
    """GCS JSON API metadata URL of a date's video_size object."""
    obj = puffer.bucket_object(day, "video_size")
    return (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quote(obj, safe='')}?"
            + urlencode(dict(fields=META_FIELDS)))


def media_url(day: str, generation: str, metageneration: str) -> str:
    """GCS JSON API media URL pinned to one object generation (and metageneration)."""
    for v in (generation, metageneration):
        if not isinstance(v, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", v):
            raise ValueError("generation/metageneration must be canonical positive integer strings")
    obj = puffer.bucket_object(day, "video_size")
    return (f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{quote(obj, safe='')}?"
            + urlencode(dict(alt="media", generation=generation, ifGenerationMatch=generation,
                             ifMetagenerationMatch=metageneration)))


def check_metadata(meta: dict, day: str) -> dict:
    """Validate GCS object metadata for a date's video_size object; returns it or raises ValueError."""
    if meta.get("bucket") != BUCKET or meta.get("name") != puffer.bucket_object(day, "video_size"):
        raise ValueError("metadata object identity mismatch")
    for key in ("size", "generation", "metageneration"):
        if not isinstance(meta.get(key), str) or not re.fullmatch(r"[1-9][0-9]{0,19}", meta[key]):
            raise ValueError(f"metadata field {key} must be a canonical positive integer string")
    if int(meta["size"]) >= OBJECT_CAP:
        raise ValueError("object larger than the per-object cap")
    try:
        md5 = base64.b64decode(meta["md5Hash"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata md5Hash required") from exc
    if len(md5) != 16 or base64.b64encode(md5).decode() != meta["md5Hash"]:
        raise ValueError("canonical base64 MD5 required")
    if meta.get("contentEncoding", "identity") not in ("identity", "gzip"):
        raise ValueError("unsupported storage encoding")
    if meta.get("contentType") not in ("text/csv", "application/octet-stream"):
        raise ValueError("unsupported content type")
    return meta


def check_media_headers(status: int, headers, meta: dict) -> None:
    """Refuse a media response whose status/headers contradict the pinned metadata (port of
    ``check_media_headers``): HTTP 200, no ``Content-Range``, ``Content-Encoding`` equal to the storage encoding,
    ``x-goog-generation`` present and equal, optional size/encoding/metageneration headers equal, and an
    ``x-goog-hash`` md5 entry (if any) equal to ``md5Hash``. Raises ValueError."""
    enc = meta.get("contentEncoding", "identity")
    if (status != 200 or headers.get("Content-Range") is not None
            or headers.get("Content-Encoding", "identity").lower() != enc):
        raise ValueError("media needs a complete HTTP 200 in the original storage encoding")
    for key, expected, required in (("Content-Length", meta["size"], False),
                                    ("x-goog-stored-content-length", meta["size"], False),
                                    ("x-goog-stored-content-encoding", enc, False),
                                    ("x-goog-generation", meta["generation"], True),
                                    ("x-goog-metageneration", meta["metageneration"], False)):
        value = headers.get(key)
        if (required and value is None) or (value is not None and value != expected):
            raise ValueError(f"media header {key}={value!r} contradicts the pinned metadata ({expected!r})")
    for entry in (headers.get("x-goog-hash") or "").split(","):
        entry = entry.strip()
        if entry.startswith("md5=") and entry[4:] != meta["md5Hash"]:
            raise ValueError("media x-goog-hash md5 contradicts the pinned metadata")


def target_path(day: str, meta: dict) -> pathlib.Path:
    """Local path of a downloaded object (gzip-stored objects keep their stored bytes)."""
    name = "video_size.csv.gz" if meta.get("contentEncoding") == "gzip" else "video_size.csv"
    return sio.confine(puffer.TEST_RAW_DIR / day / name)


def replacement_date(day: str, taken) -> str:
    """Next calendar day after ``day`` that is not in ``taken`` (pre-registered missing/corrupt-date rule)."""
    d = _dt.date.fromisoformat(day)
    while True:
        d += _dt.timedelta(days=1)
        if d.isoformat() not in taken:
            return d.isoformat()


def _tree_sha256(files) -> str:
    """SHA-256 over ``(path relative to the project root, file SHA-256)`` of every file, sorted by path."""
    h = hashlib.sha256()
    for f in sorted({pathlib.Path(f).resolve() for f in files}, key=lambda x: x.relative_to(sio.ROOT).as_posix()):
        h.update(f"{f.relative_to(sio.ROOT).as_posix()}\0{sio.file_sha256(f)}\n".encode())
    return h.hexdigest()


def analysis_code_sha256() -> str:
    """Hash of the analysis code: every file below ``CODE_DIRS`` except dot-files and dot-directories (lint and
    test caches such as ``.ruff_cache``; pre-freeze audit F1), ``__pycache__`` contents and ``*.pyc``
    (:func:`slicemech.io.source_files`)."""
    files = [f for d in CODE_DIRS for f in sio.source_files(d)]
    if not files:
        raise FileNotFoundError("no analysis code found to hash")
    return _tree_sha256(files)


def config_sha256() -> str:
    """Hash of the frozen configuration: ``CONFIG_FILES`` (all must exist) and every file matching ``CONFIG_GLOBS``."""
    missing = [str(f) for f in CONFIG_FILES if not pathlib.Path(f).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen configuration files missing: {missing}")
    files = list(CONFIG_FILES) + [f for d, pat in CONFIG_GLOBS for f in sorted(pathlib.Path(d).glob(pat))]
    return _tree_sha256(files)


def _checkpoint_path(rel: str) -> pathlib.Path:
    p = pathlib.Path(rel)
    p = (p if p.is_absolute() else sio.ROOT / p).resolve()
    if not p.is_relative_to(sio.ROOT):
        raise PermissionError(f"checkpoint {rel} lies outside the project")
    return p


def _check_test_dates(n, date_rule: str) -> int:
    """Validate the frozen TEST date count (16 or 24) and its agreement with the date rule's range; ValueError."""
    if isinstance(n, bool) or not isinstance(n, int) or n not in TEST_DATE_COUNTS:
        raise ValueError(f"test_dates must be one of {TEST_DATE_COUNTS}, got {n!r}")
    for count, end in _RULE_END.items():
        if end in str(date_rule) and count != n:
            raise ValueError(f"the date rule names the {count}-date range (ends {end}) but test_dates = {n}")
    return n


def freeze_block(checkpoints, date_rule: str | None = None, test_dates: int = 16) -> str:
    """The fenced ```` ```freeze ```` block to paste into ``prereg/PREREG.md`` at the freeze.

    Args:
        checkpoints: paths of every trained checkpoint and frozen asset (relative to the project root, or absolute
            inside it); normally ``experiments/registry.checkpoint_paths()``.
        date_rule: the pre-registered TEST date rule (one line of text); default :data:`DATE_RULES` ``[test_dates]``.
        test_dates: the pre-registered number of TEST dates, 16 or 24 (the G5 power decision, taken before the
            freeze). The CLI requires it explicitly (``--test-dates``).
    """
    ck = {}
    for c in checkpoints:
        p = _checkpoint_path(c)
        ck[p.relative_to(sio.ROOT).as_posix()] = sio.file_sha256(p)
    date_rule = DATE_RULES.get(test_dates, "") if date_rule is None else date_rule
    if not ck or not str(date_rule).strip():
        raise ValueError("a freeze needs at least one checkpoint and a date rule")
    _check_test_dates(test_dates, date_rule)
    body = dict(analysis_code_sha256=analysis_code_sha256(), config_sha256=config_sha256(), checkpoints=ck,
                date_rule=str(date_rule).strip(), test_dates=int(test_dates))
    return "```freeze\n" + yaml.safe_dump(body, sort_keys=True, allow_unicode=True, width=1 << 16) + "```\n"


def parse_freeze(text: str) -> dict:
    """The validated fields of the single ```` ```freeze ```` block of a pre-registration text (PermissionError
    when the block is missing, repeated or incomplete)."""
    blocks = _FREEZE_BLOCK.findall(text)
    if len(blocks) != 1:
        raise PermissionError(f"the frozen pre-registration needs exactly one ```freeze block, found {len(blocks)}")
    try:
        f = yaml.safe_load(blocks[0])
    except yaml.YAMLError as exc:
        raise PermissionError(f"unreadable ```freeze block: {exc}") from exc
    if not isinstance(f, dict) or any(k not in f for k in FREEZE_FIELDS):
        raise PermissionError(f"the ```freeze block needs the fields {FREEZE_FIELDS}")
    for k in ("analysis_code_sha256", "config_sha256"):
        if not isinstance(f[k], str) or not _HEX64.fullmatch(f[k]):
            raise PermissionError(f"freeze field {k} must be a lower-case hex SHA-256")
    ck = f["checkpoints"]
    if (not isinstance(ck, dict) or not ck
            or any(not isinstance(v, str) or not _HEX64.fullmatch(v) for v in ck.values())):
        raise PermissionError("freeze field checkpoints must map at least one path to its SHA-256")
    if not isinstance(f["date_rule"], str) or not f["date_rule"].strip():
        raise PermissionError("freeze field date_rule must be a non-empty text")
    try:
        _check_test_dates(f["test_dates"], f["date_rule"])
    except ValueError as exc:
        raise PermissionError(f"freeze field test_dates: {exc}") from exc
    return f


def _recorded_freeze() -> dict | None:
    m = manifest_path()
    return sio.read_json(m).get("freeze") if m.exists() else None


def _require_freeze(freeze_file, *, check_code: bool = True) -> dict:
    """Check the frozen pre-registration (module docstring) and return its provenance record
    ``dict(path, sha256, analysis_code_sha256, config_sha256)``; raises PermissionError otherwise.

    ``check_code`` compares the recorded code and config hashes with the current tree (always before a download).
    """
    p = pathlib.Path(freeze_file).resolve()
    if not p.is_relative_to(PREREG_DIR.resolve()) or not p.is_file():
        raise PermissionError(f"downloads need the frozen pre-registration file under {PREREG_DIR}")
    if "DRAFT" in p.name.upper():
        raise PermissionError("the pre-registration is still a draft")
    raw = p.read_bytes()
    draft = PREREG_DIR / "PREREG_DRAFT.md"
    if draft.is_file() and draft.resolve() != p and draft.read_bytes() == raw:
        raise PermissionError("the freeze file is a copy of the draft")
    text = raw.decode("utf-8")
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if "DRAFT" in first.upper() or "⟨DEV" in text:
        raise PermissionError("the pre-registration is still a draft (DRAFT title or unresolved ⟨DEV⟩ values)")
    f = parse_freeze(text)
    for rel, digest in f["checkpoints"].items():
        cp = _checkpoint_path(rel)
        if not cp.is_file() or sio.file_sha256(cp) != digest:
            raise PermissionError(f"checkpoint {rel} is missing or differs from its frozen SHA-256")
    if check_code:
        if f["analysis_code_sha256"] != analysis_code_sha256():
            raise PermissionError("the analysis code differs from the frozen analysis_code_sha256")
        if f["config_sha256"] != config_sha256():
            raise PermissionError("the configuration differs from the frozen config_sha256")
    sha = hashlib.sha256(raw).hexdigest()
    rec = _recorded_freeze()
    if rec is not None and rec.get("sha256") != sha:
        raise PermissionError("the freeze file differs from the one recorded at the TEST download")
    return dict(path=str(p), sha256=sha, analysis_code_sha256=f["analysis_code_sha256"],
                config_sha256=f["config_sha256"], test_dates=int(f["test_dates"]))


def _check_manifest_count(test_dates: int) -> None:
    """After the download: the fetch manifest and the effective TEST dates must have the frozen count (F5)."""
    m = manifest_path()
    if not m.exists():
        return
    extended = bool(sio.read_json(m).get("extended", False))
    n_eff = len(puffer.effective_test_dates())
    if extended != (test_dates == 24) or n_eff != test_dates:
        raise PermissionError(f"the fetch manifest covers {n_eff} TEST dates (extended={extended}) but the freeze "
                              f"fixes test_dates = {test_dates}")


DEVIATIONS_FILE = "DEVIATIONS.md"


def logged_deviation(digest: str) -> bool:
    """True when ``prereg/DEVIATIONS.md`` names ``digest`` (a post-freeze fix logged with its new hash)."""
    p = PREREG_DIR / DEVIATIONS_FILE
    return bool(_HEX64.fullmatch(str(digest))) and p.is_file() and str(digest) in p.read_text(encoding="utf-8")


def require_freeze_for_test_reads(freeze_file) -> dict:
    """Gate of every TEST read (e.g. the TEST rows of ``calibrate.data_report``, TEST window composition): the freeze
    must pass :func:`_require_freeze`. Before the download records the freeze, the code and config hashes must equal
    the frozen ones. After it, the freeze file must be the recorded one and the current code and config hashes are
    re-checked: each must equal its frozen value or be named in ``prereg/DEVIATIONS.md`` (a logged post-freeze fix).
    The record gains ``current_analysis_code_sha256``, ``current_config_sha256`` and ``deviations`` (the fields that
    differ from the freeze and are logged). After the download the number of TEST dates in the fetch manifest must
    equal the frozen ``test_dates`` (pre-freeze audit F5)."""
    recorded = _recorded_freeze() is not None
    rec = _require_freeze(freeze_file, check_code=not recorded)
    _check_manifest_count(rec["test_dates"])
    cur = dict(analysis_code_sha256=analysis_code_sha256(), config_sha256=config_sha256())
    deviations = []
    for field, digest in cur.items():
        if digest != rec[field]:
            if not logged_deviation(digest):
                raise PermissionError(
                    f"the current {field} ({digest[:12]}...) differs from the frozen one and is not logged in "
                    f"{PREREG_DIR / DEVIATIONS_FILE}: log every post-freeze change with its new hash before reading "
                    "TEST data")
            deviations.append(field)
    return dict(rec, current_analysis_code_sha256=cur["analysis_code_sha256"],
                current_config_sha256=cur["config_sha256"], deviations=deviations)


def require_test_read_gate(test_freeze=None) -> dict:
    """Gate of a TEST read that has no freeze argument of its own in the caller's API (orchestrator decision D2:
    the per-date count rule reads TEST offered traffic only when TEST windows are composed after the freeze).

    ``test_freeze`` is the frozen pre-registration file; ``None`` uses the one recorded in the fetch manifest at the
    TEST download. Either way :func:`require_freeze_for_test_reads` must pass. Before the freeze (no file given and
    none recorded) this raises PermissionError without reading any TEST source.
    """
    if test_freeze is None:
        rec = _recorded_freeze()
        if not rec or not rec.get("path"):
            raise PermissionError("TEST sources may be read only after the pre-registration freeze: no frozen "
                                  f"pre-registration is recorded in {manifest_path()} (pass test_freeze=)")
        test_freeze = rec["path"]
    return require_freeze_for_test_reads(test_freeze)


def _read_body(resp, limit: int) -> bytes:
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata response too large")
    return data


def _download_media(day: str, url: str, meta: dict, partial: pathlib.Path, opener, timeout: float) -> str:
    """Stream the pinned media to ``partial`` while hashing; returns the SHA-256 of the stored bytes. Raises
    CorruptObject on a size/EOF/MD5 mismatch (``partial`` is left in place), ValueError on header violations."""
    enc = meta.get("contentEncoding", "identity")
    req = Request(url, headers={"Accept-Encoding": enc, "User-Agent": USER_AGENT})
    size, md5, sha = int(meta["size"]), hashlib.md5(), hashlib.sha256()
    count, extra = 0, b""
    with opener(req, timeout=timeout) as resp:
        check_media_headers(resp.status, resp.headers, meta)      # before any byte is written
        with open(partial, "xb") as out:
            while count < size:
                block = resp.read(min(BLOCK, size - count))
                if not block:
                    break
                count += len(block)
                md5.update(block); sha.update(block)
                out.write(block)
            extra = resp.read(1)                   # terminal EOF probe
            if extra:
                out.write(extra)
    if count != size or extra:
        raise CorruptObject(f"{day}: got {count + len(extra)} bytes, pinned size {size}; partial kept at {partial}")
    if base64.b64encode(md5.digest()).decode() != meta["md5Hash"]:
        raise CorruptObject(f"{day}: MD5 mismatch against md5Hash; partial kept at {partial}")
    return sha.hexdigest()


def _reject(day: str, target: pathlib.Path) -> pathlib.Path:
    """Move a content-invalid object (and its metadata) aside as ``*.invalid`` and drop any cache built from it."""
    invalid = target.with_name(target.name + ".invalid")
    os.replace(target, invalid)
    meta = target.parent / "object_metadata.json"
    if meta.exists():
        os.replace(meta, meta.with_name(meta.name + ".invalid"))
    puffer.remove_cache(day)
    return invalid


def fetch_date(day: str, *, freeze_file, opener=None, timeout: float = 60.0) -> dict:
    """Download one pre-registered TEST date pinned to its object generation and check its content; returns a
    provenance record.

    ``opener(request, timeout=...)`` defaults to :func:`default_opener` (no redirects); tests pass a fake.
    Pre-freeze audit F3 (:data:`CORRUPT_CLAUSE`):

    * a size/MD5/EOF mismatch is retried once (a transient transfer error must not change the date list); the second
      mismatch raises CorruptObject and keeps the partial file for inspection;
    * the accepted object is prepared at once (``puffer.prepare_date``: decompression, every ``parse_video_size``
      validation, the recorded MD5, and at least one of the seven channels with all four formats). A failure moves
      the object to ``<target>.invalid``, removes any cache built from it and raises CorruptObject, so the
      replacement rule applies during the download, not in E0 after the manifest is final. The preparation also
      builds the TEST chunk cache one date at a time (F2).

    HTTPError from the transport, ValueError on metadata or header violations and every other error (disk, network)
    are raised unchanged. A stale ``.partial`` of an interrupted run is removed before the download (F12).
    """
    freeze = _require_freeze(freeze_file)
    if day not in puffer.test_dates(extended=True) and day not in _replacements_allowed():
        raise ValueError(f"{day} is not a pre-registered TEST date (or registered replacement)")
    opener = default_opener() if opener is None else opener
    sio.require_free_gb(3)
    req = Request(metadata_url(day), headers={"Accept-Encoding": "identity", "User-Agent": USER_AGENT})
    with opener(req, timeout=timeout) as resp:
        if resp.status != 200 or resp.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError(f"metadata needs an uncompressed HTTP 200 (got {resp.status})")
        meta = check_metadata(json.loads(_read_body(resp, 65536)), day)
    target = target_path(day, meta)
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)                # F12: a stale partial would block the resume ("xb" open)
    url = media_url(day, meta["generation"], meta["metageneration"])
    attempts = []
    for attempt in range(1, MD5_ATTEMPTS + 1):
        try:
            sha = _download_media(day, url, meta, partial, opener, timeout)
            break
        except CorruptObject as exc:
            attempts.append(str(exc))
            if attempt == MD5_ATTEMPTS:
                raise CorruptObject(f"{day}: size/MD5 check failed on {MD5_ATTEMPTS} attempts: {attempts}") from exc
            partial.unlink(missing_ok=True)
    partial.replace(target)
    sio.write_json(target.parent / "object_metadata.json", meta)
    if puffer.local_source(day).resolve() != target.resolve():
        raise RuntimeError(f"{day}: puffer.local_source resolves to {puffer.local_source(day)}, not the download")
    try:
        puffer.prepare_date(day, force=True)
    except puffer.InvalidVideoSize as exc:
        invalid = _reject(day, target)
        raise CorruptObject(f"{day}: content check failed ({exc}); object moved to {invalid}") from exc
    info = sio.read_json(puffer.cache_paths(day)[1])
    return dict(date=day, object=meta["name"], generation=meta["generation"], size=int(meta["size"]),
                md5_b64=meta["md5Hash"], sha256=sha, path=str(target), metadata_url=metadata_url(day), media_url=url,
                failed_attempts=attempts,
                content=dict(rows=info["rows"], kept_rows=info["kept_rows"], channels=info["channels"],
                             channels_missing=info["channels_missing"], verified_hashes=info["verified_hashes"]),
                freeze=freeze)


def _replacements_allowed() -> set:
    m = manifest_path()
    if m.exists():
        return set(sio.read_json(m).get("replacements", {}).values())
    return set()


def fetch_all(freeze_file, extended: bool | None = None, opener=None) -> dict:
    """Download every pre-registered TEST date, applying the replacement rule (:data:`REPLACEMENT_CLAUSE`).

    The date count comes from the freeze (``test_dates``: 16, or 24 with the extension; pre-freeze audit F5);
    ``extended`` is only a cross-check and a mismatch with the freeze, or with an existing manifest, is refused.
    A date counts as missing on HTTP 404 and as corrupt as :func:`fetch_date` defines it; any other error
    (network, disk, metadata validation) is raised unchanged so it cannot silently change the date list. Re-running
    resumes from the manifest. At the end the manifest must hold exactly the effective TEST dates (F12).
    """
    freeze = _require_freeze(freeze_file)
    n = freeze["test_dates"]
    frozen_ext = n == 24
    if extended is not None and bool(extended) != frozen_ext:
        raise ValueError(f"extended={bool(extended)} contradicts the frozen test_dates = {n}")
    dates = list(puffer.test_dates(frozen_ext))
    taken = set(puffer.test_dates(True))
    mpath = manifest_path()
    manifest = (sio.read_json(mpath) if mpath.exists()
                else dict(objects={}, replacements={}, failures={}, extended=frozen_ext))
    manifest.setdefault("freeze", freeze)            # first download records the freeze; later calls re-check it
    if bool(manifest.get("extended", False)) != frozen_ext:
        raise ValueError(f"the existing fetch manifest (extended={manifest.get('extended')}) contradicts the frozen "
                         f"test_dates = {n}")
    manifest["extended"] = frozen_ext
    taken |= set(manifest["replacements"].values())
    for day in dates:
        current = manifest["replacements"].get(day, day)
        tries = 0
        while current not in manifest["objects"]:
            try:
                manifest["objects"][current] = fetch_date(current, freeze_file=freeze_file, opener=opener)
            except (HTTPError, CorruptObject) as exc:
                if isinstance(exc, HTTPError) and exc.code != 404:
                    raise
                manifest["failures"][current] = repr(exc)
                tries += 1
                if tries > MAX_REPLACEMENTS:
                    sio.write_json(mpath, manifest)
                    raise RuntimeError(f"no replacement found for {day} after {MAX_REPLACEMENTS} failed candidates: "
                                       "the fetch stops; log the failure as a deviation in "
                                       f"{PREREG_DIR / DEVIATIONS_FILE}") from exc
                current = replacement_date(current, taken)
                taken.add(current)
                manifest["replacements"][day] = current
            sio.write_json(mpath, manifest)
    want = {manifest["replacements"].get(d, d) for d in dates}
    if set(manifest["objects"]) != want or set(puffer.split_dates("test")) != want:
        raise RuntimeError(f"the fetch manifest does not cover exactly the {n} effective TEST dates: objects "
                           f"{sorted(manifest['objects'])}, expected {sorted(want)}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--freeze", help="frozen pre-registration file under prereg/")
    ap.add_argument("--extended", action="store_true",
                    help="cross-check only: the date count comes from the freeze field test_dates (refused if 16)")
    ap.add_argument("--freeze-block", action="store_true", help="print the ```freeze block for the current tree")
    ap.add_argument("--checkpoint", action="append", default=[], help="trained checkpoint (repeat)")
    ap.add_argument("--test-dates", type=int, choices=TEST_DATE_COUNTS, default=None,
                    help="pre-registered number of TEST dates (required with --freeze-block)")
    ap.add_argument("--date-rule", default=None, help="TEST date rule (default: DATE_RULES[--test-dates])")
    a = ap.parse_args(argv)
    if a.freeze_block:
        if a.test_dates is None:
            ap.error("--freeze-block needs --test-dates 16 or 24 (the G5 decision)")
        print(freeze_block(a.checkpoint, a.date_rule, test_dates=a.test_dates), end="")
        return
    if not a.freeze:
        ap.error("--freeze is required for a download")
    m = fetch_all(a.freeze, True if a.extended else None)
    print(json.dumps(dict(objects=sorted(m["objects"]), replacements=m["replacements"]), indent=1))


if __name__ == "__main__":
    main()
