#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page


TXDOT_CITY_LIST_URL = "https://www.txdot.gov/discover/live-traffic-cameras.html"

# --- Selectors based on your grid snippet ---
GRID_ITEM_SEL = ".cctv-list-item"  # each camera card container
NAME_SEL = ".card-header span"
TS_SEL = ".card-footer span"
IMG_SEL = ".card-body img"

# Roadway filter: on TxDOT ITS pages it's typically a <select> bound to roadway choices.
# We'll try a few likely selectors and auto-detect if present.
ROADWAY_SELECT_CANDIDATES = [
    "select#roadwaySelect",
    "select[name*='roadway' i]",
    "select[id*='roadway' i]",
    "select[data-bind*='roadway' i]",
    "select",  # fallback scan
]


# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def downloads_camers_root() -> Path:
    root = Path.home() / "Downloads" / "camers"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", text).strip().replace(" ", "_")
    return (cleaned or "city").lower()


def safe_sql_identifier(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    if not slug:
        slug = "city"
    if not re.match(r"^[A-Za-z_]", slug):
        slug = "_" + slug
    return slug.lower()


def normalize_url(base: str, maybe_relative: str) -> str:
    return urljoin(base, maybe_relative)


def is_its_txdot(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower().endswith("its.txdot.gov")
    except Exception:
        return False


def looks_like_inline_snapshot(src: str) -> bool:
    return bool(src) and src.startswith("data:image") and "base64," in src


def is_placeholder_snapshot(src: str) -> bool:
    # common placeholder on ITS pages
    return bool(src) and ("NoSnapshot.png".lower() in src.lower())


def decode_data_image(src: str) -> Tuple[bytes, str]:
    """
    src like: data:image/jpeg;base64,....
    returns (bytes, ext)
    """
    header, b64data = src.split(",", 1)
    # header example: data:image/jpeg;base64
    m = re.search(r"data:image/([a-zA-Z0-9+.-]+);base64", header)
    fmt = (m.group(1).lower() if m else "jpeg")
    ext = {
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "png": ".png",
        "gif": ".gif",
        "webp": ".webp",
    }.get(fmt, ".img")
    return base64.b64decode(b64data), ext


def safe_file_part(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (text or "").strip())
    s = s.strip("_")[:max_len]
    return s or "camera"


# ----------------------------
# City list (static scrape)
# ----------------------------

@dataclass(frozen=True)
class CityLink:
    city: str
    url: str


def fetch_city_list_once() -> List[CityLink]:
    headers = {"User-Agent": "txdot-cctv-grid-scraper/2.0 (+local-script)"}
    resp = requests.get(TXDOT_CITY_LIST_URL, headers=headers, timeout=25)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    candidates: List[CityLink] = []
    for a in soup.find_all("a", href=True):
        city = (a.get_text() or "").strip()
        href = (a["href"] or "").strip()
        if not city or not href or len(city) > 45:
            continue

        url = urljoin(TXDOT_CITY_LIST_URL, href)
        ul = url.lower()

        if ("its.txdot.gov" in ul) or ("camera" in ul) or ("cameras" in ul) or ("traffic" in ul):
            if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,44}", city):
                candidates.append(CityLink(city=city, url=url))

    # Deduplicate by city; prefer /cameras link if present
    best: Dict[str, CityLink] = {}
    for c in candidates:
        k = c.city.lower()
        if k not in best:
            best[k] = c
        else:
            old = best[k]
            if ("/cameras" in c.url.lower()) and ("/cameras" not in old.url.lower()):
                best[k] = c

    return sorted(best.values(), key=lambda x: x.city.lower())


def pick_city_cli(cities: List[CityLink]) -> CityLink:
    print("\nAvailable cities (metro areas):")
    for i, c in enumerate(cities, start=1):
        print(f"  {i:2d}) {c.city}")

    while True:
        raw = input("\nSelect a city by number (or type part of the name): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(cities):
                return cities[idx - 1]
            print(f"Enter a number 1..{len(cities)}")
            continue

        needle = raw.lower()
        matches = [c for c in cities if needle in c.city.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            print("No matches. Try again.")
            continue

        print("\nMatches:")
        for i, c in enumerate(matches, start=1):
            print(f"  {i:2d}) {c.city}")
        raw2 = input("Pick a match by number: ").strip()
        if raw2.isdigit():
            idx2 = int(raw2)
            if 1 <= idx2 <= len(matches):
                return matches[idx2 - 1]
        print("Invalid selection.")


# ----------------------------
# SQLite
# ----------------------------

def ensure_sqlite_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS "{table}" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roadway TEXT,
        camera_id TEXT,
        camera_name TEXT,
        timestamp_text TEXT,
        image_ext TEXT,
        file_path TEXT,
        image_bytes BLOB,
        scraped_at_utc TEXT,
        source_url TEXT
    );
    """)
    conn.commit()


def clear_sqlite_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f'DELETE FROM "{table}";')
    conn.commit()


def insert_sqlite_row(conn: sqlite3.Connection, table: str, row: dict) -> None:
    conn.execute(
        f"""
        INSERT INTO "{table}" (
            roadway, camera_id, camera_name, timestamp_text,
            image_ext, file_path, image_bytes, scraped_at_utc, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("roadway"),
            row.get("camera_id"),
            row.get("camera_name"),
            row.get("timestamp_text"),
            row.get("image_ext"),
            row.get("file_path"),
            row.get("image_bytes"),
            row.get("scraped_at_utc"),
            row.get("source_url"),
        ),
    )


# ----------------------------
# Playwright: CCTV grid scraping
# ----------------------------

async def click_cameras_tab_if_present(page: Page) -> None:
    # From your snippet: li[title="Cameras"] a
    try:
        await page.click("li[title='Cameras'] a", timeout=15000)
    except Exception:
        pass


async def wait_for_grid(page: Page) -> None:
    # wait for any grid items to appear
    await page.wait_for_selector(GRID_ITEM_SEL, timeout=60000)


async def find_roadway_select(page: Page) -> Optional[str]:
    """
    Find the roadway filter <select> if present.
    Returns the selector string that points to the element, or None.
    """
    # Try candidates first
    for sel in ROADWAY_SELECT_CANDIDATES[:-1]:
        try:
            el = await page.query_selector(sel)
            if el:
                # Ensure it has options (more than 1 often)
                opts = await el.query_selector_all("option")
                if len(opts) >= 1:
                    return sel
        except Exception:
            continue

    # Fallback: scan all selects and pick one that looks like roadway filter
    selects = await page.query_selector_all("select")
    for i, sel_el in enumerate(selects):
        try:
            options = await sel_el.query_selector_all("option")
            if len(options) < 2:
                continue
            texts = []
            for o in options[:10]:
                t = (await o.inner_text()) or ""
                texts.append(t.lower())
            joined = " ".join(texts)
            # heuristic: roadway filters often include "IH", "US", "FM", "Loop", etc.
            if any(k in joined for k in ["ih", "us", "fm", "loop", "spur", "sh"]):
                # Build a stable selector via nth-of-type
                return f"select:nth-of-type({i+1})"
        except Exception:
            continue

    return None


async def roadway_options(page: Page, select_sel: str) -> List[Tuple[str, str]]:
    """
    Returns list of (value, label) for roadway dropdown.
    """
    sel = await page.query_selector(select_sel)
    if not sel:
        return []
    opts = await sel.query_selector_all("option")
    out = []
    for o in opts:
        value = (await o.get_attribute("value")) or ""
        label = ((await o.inner_text()) or "").strip()
        if not label:
            continue
        out.append((value, label))
    # dedupe by value/label
    seen = set()
    uniq = []
    for v, l in out:
        k = (v, l.lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append((v, l))
    return uniq


async def select_roadway(page: Page, select_sel: str, value: str) -> None:
    # Selecting triggers knockout updates; wait a moment after selection + grid items refresh
    await page.select_option(select_sel, value=value)
    # allow UI to update
    await page.wait_for_timeout(750)


async def scrape_grid_once(page: Page, base_url: str, roadway_label: str, scraped_at: str, images_dir: Path) -> List[dict]:
    """
    Scrape ONLY the CCTV grid cards currently displayed.
    - camera_id from div.cctv-list-item[id]
    - camera_name from header span
    - timestamp from footer span
    - image is base64 data URL (decode & save)
    """
    await wait_for_grid(page)
    items = await page.query_selector_all(GRID_ITEM_SEL)

    rows: List[dict] = []
    for idx, item in enumerate(items, start=1):
        camera_id = await item.get_attribute("id")

        name_el = await item.query_selector(NAME_SEL)
        camera_name = ((await name_el.inner_text()).strip() if name_el else (camera_id or "Unknown Camera"))

        ts_el = await item.query_selector(TS_SEL)
        timestamp_text = ((await ts_el.inner_text()).strip() if ts_el else None)

        img_el = await item.query_selector(IMG_SEL)
        img_src = (await img_el.get_attribute("src")) if img_el else None

        image_bytes = None
        image_ext = None
        file_path = None

        if img_src and not is_placeholder_snapshot(img_src):
            # If it's inline base64 data URL, decode it
            if looks_like_inline_snapshot(img_src):
                image_bytes, image_ext = decode_data_image(img_src)
            else:
                # Some pages might provide a normal URL. We won't fetch it (you asked for grid images),
                # but we can still store the URL and skip bytes.
                image_ext = None
                image_bytes = None

        # Save image file if we have bytes
        if image_bytes:
            safe_name = safe_file_part(camera_name)
            # include roadway in filename to prevent collisions
            safe_road = safe_file_part(roadway_label, max_len=40)
            filename = f"{safe_road}_{idx:04d}_{safe_name}{image_ext}"
            path = images_dir / filename
            path.write_bytes(image_bytes)
            file_path = str(path)

        rows.append({
            "roadway": roadway_label,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "timestamp_text": timestamp_text,
            "image_ext": image_ext,
            "file_path": file_path,
            "image_bytes": image_bytes,
            "scraped_at_utc": scraped_at,
            "source_url": base_url,
        })

    return rows


# ----------------------------
# Output: CSV + JSON
# ----------------------------

def save_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "roadway",
            "camera_id",
            "camera_name",
            "timestamp_text",
            "image_ext",
            "file_path",
            "scraped_at_utc",
            "source_url",
        ])
        for r in rows:
            w.writerow([
                r.get("roadway", ""),
                r.get("camera_id", ""),
                r.get("camera_name", ""),
                r.get("timestamp_text", ""),
                r.get("image_ext", ""),
                r.get("file_path", ""),
                r.get("scraped_at_utc", ""),
                r.get("source_url", ""),
            ])


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ----------------------------
# Main
# ----------------------------

async def run() -> int:
    # Step 1: pick city from TxDOT list
    cities = fetch_city_list_once()
    if not cities:
        print("ERROR: Could not extract cities.")
        return 3

    chosen = pick_city_cli(cities)

    # Setup output paths
    root = downloads_camers_root()
    city_slug = safe_slug(chosen.city)
    table_name = safe_sql_identifier(city_slug)

    city_dir = root / city_slug
    images_dir = city_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = city_dir / f"{city_slug}_cameras.csv"
    json_path = city_dir / f"{city_slug}_cameras.json"
    db_path = root / "camers.db"

    scraped_at = utc_now_iso()

    print(f"\nSelected: {chosen.city}")
    print(f"URL:      {chosen.url}")
    print(f"Folder:   {city_dir}")
    print(f"Images:   {images_dir}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"SQLite:   {db_path} (table: {table_name})\n")

    all_rows: List[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(chosen.url, timeout=60000, wait_until="domcontentloaded")
        await click_cameras_tab_if_present(page)

        # Wait for grid to appear at least once
        await wait_for_grid(page)

        base_url = page.url  # after redirects etc.

        # Step 2: iterate roadway filter options (if available)
        select_sel = await find_roadway_select(page)

        if select_sel:
            opts = await roadway_options(page, select_sel)
            # Some selects have a placeholder option; still iterate all, but skip empty values if needed.
            print(f"Roadway filter found: {select_sel} with {len(opts)} options")

            # Iterate options; for each, select and scrape the grid
            for value, label in opts:
                if value is None:
                    continue
                try:
                    await select_roadway(page, select_sel, value)
                    # Wait for any grid items to load/update
                    await wait_for_grid(page)
                    rows = await scrape_grid_once(page, base_url, label, scraped_at, images_dir)
                    all_rows.extend(rows)
                    print(f"  - {label}: {len(rows)} grid items")
                except Exception as e:
                    print(f"  - {label}: ERROR ({e})")
        else:
            # No roadway filter; just scrape once
            print("No roadway filter detected; scraping current grid only.")
            rows = await scrape_grid_once(page, base_url, "All", scraped_at, images_dir)
            all_rows.extend(rows)
            print(f"  - All: {len(rows)} grid items")

        await browser.close()

    # Step 3: de-duplicate by (roadway, camera_id, timestamp) to keep output clean
    seen = set()
    deduped: List[dict] = []
    for r in all_rows:
        key = (
            (r.get("roadway") or "").lower(),
            (r.get("camera_id") or "").lower(),
            (r.get("timestamp_text") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # Step 4: save CSV/JSON
    save_csv(csv_path, deduped)

    payload = {
        "source_city_list": TXDOT_CITY_LIST_URL,
        "selected_city": {"city": chosen.city, "url": chosen.url},
        "scraped_at_utc": scraped_at,
        "total_rows_before_dedupe": len(all_rows),
        "total_rows_after_dedupe": len(deduped),
        "images_folder": str(images_dir),
        "sqlite": {"db_path": str(db_path), "table": table_name},
        "cameras": [
            {k: v for k, v in r.items() if k != "image_bytes"} for r in deduped
        ],
        "note": "Images are taken ONLY from the CCTV grid cards. Inline base64 snapshots are decoded and saved.",
    }
    save_json(json_path, payload)

    # Step 5: SQLite save (BLOB + file path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_sqlite_table(conn, table_name)
        clear_sqlite_table(conn, table_name)
        for r in deduped:
            insert_sqlite_row(conn, table_name, r)
        conn.commit()
    finally:
        conn.close()

    saved_images = sum(1 for r in deduped if r.get("file_path"))
    print(f"\nDone.")
    print(f"Grid rows saved: {len(deduped)}")
    print(f"Images saved:    {saved_images}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DB:   {db_path} (table: {table_name})")
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
