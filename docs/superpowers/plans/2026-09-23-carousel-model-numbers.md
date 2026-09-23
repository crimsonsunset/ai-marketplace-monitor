# Read model numbers from every carousel photo

**Issue:** [#1](https://github.com/crimsonsunset/ai-marketplace-monitor/issues/1)
**Branch:** `feature/1-carousel-photo-model-numbers`
**Base:** `session-persistence` @ `a816333` (stacked on [PR #2](https://github.com/crimsonsunset/ai-marketplace-monitor/pull/2))
**Date:** 2026-09-23
**Status:** Implemented, AC met

## Flow state

| Field | Value |
|---|---|
| Gate | 6 (CI green, no bot review) |
| Ticket | #1 |
| Branch | feature/1-carousel-photo-model-numbers |
| Repos | ai-marketplace-monitor (fork `crimsonsunset/ai-marketplace-monitor`) |
| Isolation | in-place |
| Base | session-persistence @ a816333 |
| QA mode | pytest |
| Gates | all |
| Updated | 2026-09-23 |

## Overview

On a Facebook item page, click through the photo gallery, download each slide inside the logged-in Playwright session, and ask `google/gemini-3.1-flash-lite` for a sticker model. Store `brand`, `model`, and `size_in` on the listing and show them in messenger messages, email, and the found CSV.

This does not replace the text deal-rating prompt. It does not OCR the search-card thumbnail. It does not treat a series name in the description ("43 Class C350 Series") as a sticker SKU.

The branch is cut from `session-persistence` because that branch already moves `get_image_url` and the listing parser. Fork `main` is behind upstream and does not have those changes.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where the JSON lands | New optional fields on `Listing`, then messenger, email, and the found CSV | Answered at Gate 1. The cache alone would hide the model from the only places a find is read. |
| How images are collected | Click through the gallery in the logged-in browser | Answered at Gate 1. Later slides are not the first `img`. A static dump misses slides that are not in the DOM yet. |
| Hero image | `Listing.image` stays one URL for the email attachment | Email already fetches that URL with `requests`. The carousel set is a separate field used only for this call. |
| Which page | Item page only, all item-page layouts | The sticker is not the search thumbnail. Search-card OCR is out of scope in the issue. |
| Download | `page.context.request` while the item page is still open | Facebook CDN URLs 403 outside the session and they expire. `fetch_with_retry` is unauthenticated. |
| Model | `google/gemini-3.1-flash-lite` via OpenRouter | Named in the issue. OpenAI-compatible client, `base_url` `https://openrouter.ai/api/v1`. Same shape as `OpenAIBackend`, separate call. |
| Key | `OPENROUTER_KEY_AIMM` from the environment | Do not commit it. Missing key: log and skip the call. |
| Thinking | Minimal | Extraction, not reasoning. Output tokens are the expensive part. |
| Resize | Long side about 1600 | `resize_image_data` caps at 800x600 for email and will blur a sticker. New parameter or sibling helper. Do not change the email cap. |
| Skip | Title or description already contains a full model token | Series wording does not count. Skip means no API call. |
| Cache | New `CacheType` keyed by a hash of the image bytes | Not `Listing.hash`. A photo change must not re-rate the deal. |
| Hash | Exclude the new fields and the carousel URL list from `Listing.hash`, same as `image` | `hash_dict(asdict(self))` would otherwise fold them in. |
| Cached details | Do not backfill listings already in `LISTING_DETAILS` | The page is gone. Extraction runs on a fresh item-page parse. |
| Rating prompt | Unchanged | Issue out of scope. Do not inject the JSON into `get_prompt()`. |

## Scope

In:

- Click next on the item-page gallery and keep each distinct listing-image URL.
- Download and resize those bytes in-session, then one vision call for the set.
- Optional `brand`, `model`, `size_in`, `image_index`, `confidence` on `Listing`, with defaults so old cache dicts still load in `Listing.from_cache`.
- Copy those fields from the parsed detail onto the search listing. Today the merge only copies `condition`, `seller`, and `description`, so an item-page image never reaches the object that gets notified.
- A model line in plain, markdown, and HTML push messages, and in the email body. Omit the line when brand and model are both empty.
- `brand`, `model`, `size_in` columns on the found CSV. Empty for old rows.
- Tests for the hash exclusion, the skip rule, the gallery walker against a fake page, and the JSON parser.

Out:

- Replacing or editing the text deal-rating prompt. The issue says the rating path stays.
- OCR on the search-card thumbnail. The sticker is not that image.
- Guessing a model from a series name. That stays a separate text path.
- Re-opening cached listings to backfill models. No page, no session, no call.
- Changing the email attachment. It keeps the single hero URL.
- Other marketplaces. `MarketPlace` is Facebook only.

## Architecture

`Listing` gains optional fields with defaults (`""` or `None`, and `images: list[str]` defaulting to empty). `from_cache` does `cls(**cached_dict)`. Required new fields would break every saved listing.

`Listing.hash` keeps excluding `image`, and also excludes `images`, `brand`, `model`, `size_in`, `image_index`, and `confidence`.

Flow for a listing that is not already in `LISTING_DETAILS`:

1. `get_listing_details` opens the item page and `parse_listing` runs, as it does now.
2. A shared walker on `FacebookItemPage` clicks through the gallery, records each distinct CDN image URL, and stops when the next control is gone, the URL repeats, or it has seen 8 slides. `get_image_url` still returns the hero for `Listing.image`.
3. Before the page is left, download each URL with `context.request`. Resize so the long side is about 1600.
4. If the title or description already has a full model token, skip the call and leave the model fields empty.
5. Otherwise hash the bytes. On a cache hit, reuse the stored JSON. On a miss, send the images plus title and description to OpenRouter. Ask for JSON only: `brand`, `model`, `size_in`, `image_index`, `confidence`. Null when nothing is printed. Do not invent a SKU suffix.
6. Write the result into the new `CacheType` and onto the `Listing`, then `to_cache` as today.
7. The search loop copies the new fields onto the listing it yields, next to the existing `condition` / `seller` / `description` copy.
8. Push `notify` and the email body add one model line. `found_export` adds three columns. The deal-rating `AIResponse` is untouched.

OpenRouter request: OpenAI client, model `google/gemini-3.1-flash-lite`, image parts are the resized bytes (not CDN URLs), `extra_body` reasoning effort `minimal`. Headers can follow the existing `X-Title: AI Marketplace Monitor` pattern. The call is not an `[ai.*]` rating agent and is not selected by `item.ai`.

## Files

| File | Change |
|---|---|
| [listing.py](src/ai_marketplace_monitor/listing.py) | Optional model fields and `images`. Exclude them from `hash`. Defaults so old cache rows still construct. |
| [utils.py](src/ai_marketplace_monitor/utils.py) | New `CacheType` for the vision result. Long-side resize that does not change the 800x600 email default. |
| [facebook.py](src/ai_marketplace_monitor/facebook.py) | Gallery walker on the item page. In-session download. Merge the new fields onto the search listing. `login()` asserts `page` is set so mypy is clean on the session-persistence login path. |
| [ai.py](src/ai_marketplace_monitor/ai.py) | Vision call and JSON parse. Skip rule. Cache get/set by image-bytes hash. No change to `get_prompt` or rating parse. |
| [notification.py](src/ai_marketplace_monitor/notification.py) | Model line in plain, markdown, and HTML messages. |
| [email.html.j2](src/ai_marketplace_monitor/email.html.j2) | Same model line in the HTML email body. Hero image markup stays. |
| [email_notify.py](src/ai_marketplace_monitor/email_notify.py) | Same model line in the plain-text email body. The attachment is still `listing.image` only. |
| [found_export.py](src/ai_marketplace_monitor/webui/found_export.py) | `brand`, `model`, `size_in` columns, filled from the listing dict. |
| [tests/test_facebook.py](tests/test_facebook.py) | Fake-page walker: N distinct slides, stop on repeat, ignore non-listing icons. |
| New `tests/test_model_vision.py` | Hash exclusion, skip rule (token vs series name), JSON parse, cache hit does not call. |
| [CHANGELOG.md](CHANGELOG.md) | Unreleased note. |

## Phasing

### Phase 1: Listing fields and hash

About an hour.

- Add the optional fields and the `images` list with defaults.
- Exclude the new keys from `Listing.hash`.
- Confirm `Listing.from_cache` still loads a dict that has none of the new keys.

**Outcome:** A test builds a listing, changes `model` and `images`, and the hash stays the same. A dict shaped like today's cache still becomes a `Listing`.

### Phase 2: Gallery walk and merge

About half a day.

- Click through the item-page gallery and collect distinct listing-image URLs.
- Leave `get_image_url` as the hero.
- Copy `images` and the model fields from the parsed detail onto the yielded search listing.

**Outcome:** A fake page with three slides and a next control returns three URLs and then stops. A search-merge test shows those URLs on the listing that gets yielded, not only on the detail object that is discarded today.

### Phase 3: Download, resize, skip, call, cache

About half a day.

- Download with the Playwright context while the item page is open.
- Resize to a long side of about 1600. Leave the email helper's 800x600 default alone.
- Skip when a full model token is already in the title or description. "43 Class C350 Series" does not skip.
- Call OpenRouter only on a cache miss. Key is `OPENROUTER_KEY_AIMM`. Missing key skips the call.
- Store the JSON on the listing and under the new cache type. Do not call when details came from `LISTING_DETAILS`.

**Outcome:** Tests cover the skip rule, a cache hit that does not call the client, and a JSON body that fills `brand`, `model`, `size_in`, `image_index`, and `confidence`. `Listing.hash` is still unchanged.

### Phase 4: Notifications, CSV, changelog

About a couple of hours.

- Add the model line to plain, markdown, and HTML push messages, and to the email body. No line when brand and model are empty.
- Add the three CSV columns.
- Changelog entry under Unreleased.

**Outcome:** A notification test shows the model line when the fields are set and the old message when they are empty. A found-CSV test includes `brand`, `model`, and `size_in`. The email still attaches only `listing.image`.

## Key files referenced

| File | Note |
|---|---|
| [facebook.py](src/ai_marketplace_monitor/facebook.py) | `get_image_url` is the first `img`. Search merge at the `condition` / `seller` / `description` loop drops the item-page image. |
| [listing.py](src/ai_marketplace_monitor/listing.py) | `image: str`. Hash skips `image` only. `from_cache` splats the cached dict. |
| [ai.py](src/ai_marketplace_monitor/ai.py) | Text rating only. OpenAI-compatible client. No image parts, no JSON schema. |
| [utils.py](src/ai_marketplace_monitor/utils.py) | `CacheType` has no vision tag. `resize_image_data` is 800x600. |
| [notification.py](src/ai_marketplace_monitor/notification.py) | Message bodies are title, price, location, URL, description, and the rating comment. |
| [found_export.py](src/ai_marketplace_monitor/webui/found_export.py) | `CSV_COLUMNS` has no model fields. Rows come from the cached listing dict. |
| [email_notify.py](src/ai_marketplace_monitor/email_notify.py) | Fetches `listing.image` with `requests`, then the 800x600 resize. |

## Related

- Issue: [#1](https://github.com/crimsonsunset/ai-marketplace-monitor/issues/1)
- Base branch PR: [#2](https://github.com/crimsonsunset/ai-marketplace-monitor/pull/2)
- Prior plan in this repo: [2026-07-16-export-found-csv.md](2026-07-16-export-found-csv.md)
