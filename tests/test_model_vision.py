"""Tests for carousel model reads: hash, skip rule, gallery walk, and JSON."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

from diskcache import Cache  # type: ignore
from PIL import Image

from ai_marketplace_monitor.ai import (
    AIResponse,
    has_full_model_token,
    parse_model_json,
    read_model_from_photos,
)
from ai_marketplace_monitor.email_notify import EmailNotificationConfig
from ai_marketplace_monitor.facebook import copy_item_page_fields, walk_gallery
from ai_marketplace_monitor.listing import Listing
from ai_marketplace_monitor.notification import (
    NotificationStatus,
    PushNotificationConfig,
    format_model_line,
)
from ai_marketplace_monitor.utils import CacheType, resize_image_data, resize_image_long_side


def _listing(**overrides: str) -> Listing:
    """Build a listing with the fields a search card would have."""
    listing = Listing(
        marketplace="facebook",
        name="tv",
        id="999",
        title='Toshiba 43" TV',
        image="https://scontent.xx.fbcdn.net/v/t1/hero.jpg",
        price="$80",
        post_url="https://www.facebook.com/marketplace/item/999",
        location="Austin, TX",
        seller="sam",
        condition="used",
        description="43 Class C350 Series",
    )
    for key, value in overrides.items():
        setattr(listing, key, value)
    return listing


def _jpeg(width: int, height: int) -> bytes:
    """Return a solid JPEG of the given size."""
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_hash_ignores_model_and_images() -> None:
    base = _listing()
    changed = _listing()
    changed.image = "https://scontent.xx.fbcdn.net/v/t1/other.jpg"
    changed.images = ["https://scontent.xx.fbcdn.net/v/t1/slide.jpg"]
    changed.brand = "Toshiba"
    changed.model = "UN55TU7000"
    changed.size_in = "55"
    changed.image_index = 2
    changed.confidence = "high"
    assert base.hash == changed.hash


def test_from_cache_loads_old_rows(tmp_path: Path) -> None:
    local = Cache(str(tmp_path / "cache"))
    url = "https://www.facebook.com/marketplace/item/999"
    local.set(
        (CacheType.LISTING_DETAILS.value, url.split("?")[0]),
        {
            "marketplace": "facebook",
            "name": "tv",
            "id": "999",
            "title": 'Toshiba 43" TV',
            "image": "https://example.com/a.jpg",
            "price": "$80",
            "post_url": url,
            "location": "Austin, TX",
            "seller": "sam",
            "condition": "used",
            "description": "43 Class C350 Series",
        },
        tag=CacheType.LISTING_DETAILS.value,
    )
    loaded = Listing.from_cache(url, local_cache=local)
    assert loaded is not None
    assert loaded.brand == ""
    assert loaded.model == ""
    assert loaded.images == []
    local.close()


def test_series_name_is_not_a_model_token() -> None:
    assert has_full_model_token('Toshiba 43" TV', "43 Class C350 Series") is False


def test_full_model_token_skips() -> None:
    assert has_full_model_token("Samsung UN55TU7000", "") is True


def test_walk_gallery_three_slides_then_stop() -> None:
    slides = [
        "https://scontent.xx.fbcdn.net/v/t1/a.jpg?token=1",
        "https://scontent.xx.fbcdn.net/v/t1/b.jpg?token=1",
        "https://static.xx.fbcdn.net/rsrc.php/play_48dp.png",
        "https://scontent.xx.fbcdn.net/v/t1/c.jpg?token=1",
    ]
    index = {"i": 0}

    def current_url() -> str:
        return slides[index["i"]]

    def has_next() -> bool:
        return index["i"] < len(slides) - 1

    def click_next() -> None:
        index["i"] += 1

    urls = walk_gallery(current_url, has_next, click_next)
    assert urls == [
        "https://scontent.xx.fbcdn.net/v/t1/a.jpg?token=1",
        "https://scontent.xx.fbcdn.net/v/t1/b.jpg?token=1",
        "https://scontent.xx.fbcdn.net/v/t1/c.jpg?token=1",
    ]


def test_walk_gallery_stops_when_the_slide_repeats() -> None:
    slides = [
        "https://scontent.xx.fbcdn.net/v/t1/a.jpg?token=1",
        "https://scontent.xx.fbcdn.net/v/t1/b.jpg?token=2",
    ]
    index = {"i": 0}

    def current_url() -> str:
        return slides[index["i"] % len(slides)]

    def has_next() -> bool:
        return True

    def click_next() -> None:
        index["i"] += 1

    urls = walk_gallery(current_url, has_next, click_next)
    assert urls == slides


def test_copy_item_page_fields_keeps_the_hero_image() -> None:
    search = _listing()
    search.image = "https://scontent.xx.fbcdn.net/v/t1/card.jpg"
    search.condition = ""
    details = _listing()
    details.image = "https://scontent.xx.fbcdn.net/v/t1/hero.jpg"
    details.images = [
        "https://scontent.xx.fbcdn.net/v/t1/hero.jpg",
        "https://scontent.xx.fbcdn.net/v/t1/sticker.jpg",
    ]
    details.brand = "Toshiba"
    details.model = "43C350U"
    details.condition = "used"
    copy_item_page_fields(search, details)
    assert search.image == "https://scontent.xx.fbcdn.net/v/t1/card.jpg"
    assert search.images == details.images
    assert search.model == "43C350U"
    assert search.condition == "used"


def test_resize_long_side_caps_at_1600_and_email_box_is_unchanged() -> None:
    large = _jpeg(2000, 1000)
    resized = resize_image_long_side(large)
    with Image.open(BytesIO(resized)) as image:
        assert max(image.size) <= 1600
        assert max(image.size) > 800

    email = resize_image_data(_jpeg(1000, 1000))
    with Image.open(BytesIO(email)) as image:
        assert image.size[0] <= 800
        assert image.size[1] <= 600


def test_parse_model_json_strips_fences_and_nulls() -> None:
    payload = parse_model_json(
        '```json\n{"brand": "Toshiba", "model": null, "size_in": "43", '
        '"image_index": 2, "confidence": "high"}\n```'
    )
    assert payload == {
        "brand": "Toshiba",
        "model": "",
        "size_in": "43",
        "image_index": 2,
        "confidence": "high",
    }


def test_cache_hit_does_not_call_again(tmp_path: Path) -> None:
    local = Cache(str(tmp_path / "cache"))
    calls = {"n": 0}

    def complete(parts: List[Dict[str, object]]) -> str:
        calls["n"] += 1
        assert any(part.get("type") == "image_url" for part in parts)
        return (
            '{"brand":"Toshiba","model":"UN55TU7000","size_in":"55",'
            '"image_index":2,"confidence":"high"}'
        )

    first = _listing()
    blobs = [_jpeg(32, 32)]
    read_model_from_photos(first, blobs, local_cache=local, complete=complete)
    assert first.model == "UN55TU7000"
    assert first.image_index == 2
    assert calls["n"] == 1

    second = _listing()
    read_model_from_photos(second, blobs, local_cache=local, complete=complete)
    assert second.model == "UN55TU7000"
    assert calls["n"] == 1
    assert first.hash == second.hash
    local.close()


def test_push_and_email_include_the_model_line() -> None:
    listing = _listing(brand="Toshiba", model="UN55TU7000", size_in="55")
    rating = AIResponse(score=4, comment="Fine set")
    assert "\nModel: Toshiba UN55TU7000, 55 in" == format_model_line(listing, "plain_text")
    assert format_model_line(_listing(), "plain_text") == ""

    notifier = PushNotificationConfig(name="push", message_format="plain_text")
    sent: List[str] = []

    def capture(title: str, message: str, logger: object = None) -> bool:
        sent.append(message)
        return True

    notifier.send_message = capture  # type: ignore[method-assign]
    assert notifier.notify([listing], [rating], [NotificationStatus.NOT_NOTIFIED])
    assert "Model: Toshiba UN55TU7000, 55 in" in sent[0]

    blank = _listing()
    sent.clear()
    assert notifier.notify([blank], [rating], [NotificationStatus.NOT_NOTIFIED])
    assert "Model:" not in sent[0]

    email = EmailNotificationConfig(name="mail")
    text = email.get_text_message([listing], [rating], [NotificationStatus.NOT_NOTIFIED])
    assert "Model: Toshiba UN55TU7000, 55 in" in text
    html, images = email.get_html_message(
        [_listing(brand="Toshiba", model="UN55TU7000", size_in="55", image="")],
        [rating],
        [NotificationStatus.NOT_NOTIFIED],
    )
    assert images == []
    assert "Toshiba" in html
    assert "UN55TU7000" in html
