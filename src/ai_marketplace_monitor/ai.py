import base64
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from logging import Logger
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Type, TypeVar

from diskcache import Cache  # type: ignore
from openai import OpenAI  # type: ignore
from rich.pretty import pretty_repr

from .listing import Listing
from .marketplace import TItemConfig, TMarketplaceConfig
from .utils import BaseConfig, CacheType, CounterItem, cache, counter, hilight


class AIServiceProvider(Enum):
    OPENAI = "OpenAI"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    ANTHROPIC = "Anthropic"
    OLLAMA = "Ollama"


@dataclass
class AIResponse:
    score: int
    comment: str
    name: str = ""
    # True when this response was loaded from a previous run's cache rather
    # than freshly computed. Not meaningful as stored cache content — always
    # recomputed on load, see from_cache().
    cached: bool = False

    NOT_EVALUATED: ClassVar = "Not evaluated by AI"

    @property
    def conclusion(self: "AIResponse") -> str:
        return {
            1: "No match",
            2: "Potential match",
            3: "Poor match",
            4: "Good match",
            5: "Great deal",
        }[self.score]

    @property
    def style(self: "AIResponse") -> str:
        if self.comment == self.NOT_EVALUATED:
            return "dim"
        if self.score < 3:
            return "fail"
        if self.score > 3:
            return "succ"
        return "name"

    @property
    def stars(self: "AIResponse") -> str:
        full_stars = self.score
        empty_stars = 5 - full_stars
        return (
            '<span style="color: #FFD700; font-size: 20px;">★</span>' * full_stars
            + '<span style="color: #D3D3D3; font-size: 20px;">☆</span>' * empty_stars
        )

    @classmethod
    def from_cache(
        cls: Type["AIResponse"],
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
        local_cache: Cache | None = None,
    ) -> Optional["AIResponse"]:
        res = (cache if local_cache is None else local_cache).get(
            (CacheType.AI_INQUIRY.value, item_config.hash, marketplace_config.hash, listing.hash)
        )
        if res is None:
            return None
        return AIResponse(**{**res, "cached": True})

    def to_cache(
        self: "AIResponse",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
        local_cache: Cache | None = None,
    ) -> None:
        (cache if local_cache is None else local_cache).set(
            (CacheType.AI_INQUIRY.value, item_config.hash, marketplace_config.hash, listing.hash),
            asdict(self),
            tag=CacheType.AI_INQUIRY.value,
        )


@dataclass
class AIConfig(BaseConfig):
    # this argument is required

    api_key: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    max_retries: int = 10
    timeout: int | None = None

    def handle_provider(self: "AIConfig") -> None:
        if self.provider is None:
            return
        if self.provider.lower() not in [x.value.lower() for x in AIServiceProvider]:
            raise ValueError(
                f"""AIConfig requires a valid service provider. Valid providers are {hilight(", ".join([x.value for x in AIServiceProvider]))}"""
            )

    def handle_api_key(self: "AIConfig") -> None:
        if self.api_key is None:
            return
        if not isinstance(self.api_key, str):
            raise ValueError("AIConfig requires a string api_key.")
        self.api_key = self.api_key.strip()

    def handle_max_retries(self: "AIConfig") -> None:
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("AIConfig requires a positive integer max_retries.")

    def handle_timeout(self: "AIConfig") -> None:
        if self.timeout is None:
            return
        if not isinstance(self.timeout, int) or self.timeout < 0:
            raise ValueError("AIConfig requires a positive integer timeout.")


@dataclass
class OpenAIConfig(AIConfig):
    def handle_api_key(self: "OpenAIConfig") -> None:
        if self.api_key is None:
            raise ValueError("OpenAI requires a string api_key.")


@dataclass
class DeekSeekConfig(OpenAIConfig):
    pass


@dataclass
class GeminiConfig(OpenAIConfig):
    pass


@dataclass
class OllamaConfig(OpenAIConfig):
    api_key: str | None = field(default="ollama")  # required but not used.

    def handle_base_url(self: "OllamaConfig") -> None:
        if self.base_url is None:
            raise ValueError("Ollama requires a string base_url.")

    def handle_model(self: "OllamaConfig") -> None:
        if self.model is None:
            raise ValueError("Ollama requires a string model.")


@dataclass
class AnthropicConfig(AIConfig):
    def handle_api_key(self: "AnthropicConfig") -> None:
        if self.api_key is None:
            raise ValueError("Anthropic requires a string api_key.")


TAIConfig = TypeVar("TAIConfig", bound=AIConfig)


class AIBackend(Generic[TAIConfig]):
    def __init__(self: "AIBackend", config: AIConfig, logger: Logger | None = None) -> None:
        self.config = config
        self.logger = logger
        self.client: Any = None

    @classmethod
    def get_config(cls: Type["AIBackend"], **kwargs: Any) -> TAIConfig:
        raise NotImplementedError("get_config method must be implemented by subclasses.")

    def connect(self: "AIBackend") -> None:
        raise NotImplementedError("Connect method must be implemented by subclasses.")

    def get_prompt(
        self: "AIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> str:
        prompt = (
            f"""A user wants to buy a {item_config.name} from Facebook Marketplace. """
            f"""Search phrases: "{'" and "'.join(item_config.search_phrases)}", """
        )
        if item_config.description:
            prompt += f"""Description: "{item_config.description}", """
        #
        max_price = item_config.max_price or 0
        min_price = item_config.min_price or 0
        if max_price and min_price:
            prompt += f"""Price range: {min_price} to {max_price}. """
        elif max_price:
            prompt += f"""Max price {max_price}. """
        elif min_price:
            prompt += f"""Min price {min_price}. """
        #
        if item_config.antikeywords:
            prompt += f"""Exclude keywords "{'" and "'.join(item_config.antikeywords)}" in title or description."""
        #
        prompt += (
            f"""\n\nThe user found a listing titled "{listing.title}" in {listing.condition} condition, """
            f"""priced at {listing.price}, located in {listing.location}, """
            f"""posted at {listing.post_url} with description "{listing.description}"\n\n"""
        )
        # prompt
        if item_config.prompt is not None:
            prompt += item_config.prompt
        elif marketplace_config.prompt is not None:
            prompt += marketplace_config.prompt
        else:
            prompt += (
                "Evaluate how well this listing matches the user's criteria. Assess the description, MSRP, model year, "
                "condition, and seller's credibility."
            )
        # extra_prompt
        prompt += "\n"
        if item_config.extra_prompt is not None:
            prompt += f"\n{item_config.extra_prompt.strip()}\n"
        elif marketplace_config.extra_prompt is not None:
            prompt += f"\n{marketplace_config.extra_prompt.strip()}\n"
        # rating_prompt
        if item_config.rating_prompt is not None:
            prompt += f"\n{item_config.rating_prompt.strip()}\n"
        elif marketplace_config.rating_prompt is not None:
            prompt += f"\n{marketplace_config.rating_prompt.strip()}\n"
        else:
            prompt += (
                "\nRate from 1 to 5 based on the following: \n"
                "1 - No match: Missing key details, wrong category/brand, or suspicious activity (e.g., external links).\n"
                "2 - Potential match: Lacks essential info (e.g., condition, brand, or model); needs clarification.\n"
                "3 - Poor match: Some mismatches or missing details; acceptable but not ideal.\n"
                "4 - Good match: Mostly meets criteria with clear, relevant details.\n"
                "5 - Great deal: Fully matches criteria, with excellent condition or price.\n"
                "Conclude with:\n"
                '"Rating <1-5>: <summary>"\n'
                "where <1-5> is the rating and <summary> is a brief recommendation (max 30 words)."
            )
        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Prompt]", "info")} {prompt}""")
        return prompt

    def evaluate(
        self: "AIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        raise NotImplementedError("Confirm method must be implemented by subclasses.")


class OpenAIBackend(AIBackend):
    default_model = "gpt-4o"
    # the default is f"https://api.openai.com/v1"
    base_url: str | None = None

    @classmethod
    def get_config(cls: Type["OpenAIBackend"], **kwargs: Any) -> OpenAIConfig:
        return OpenAIConfig(**kwargs)

    def connect(self: "OpenAIBackend") -> None:
        if self.client is None:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or self.base_url,
                timeout=self.config.timeout,
                default_headers={
                    "X-Title": "AI Marketplace Monitor",
                    "HTTP-Referer": "https://github.com/BoPeng/ai-marketplace-monitor",
                },
            )
            if self.logger:
                self.logger.info(f"""{hilight("[AI]", "name")} {self.config.name} connected.""")

    def evaluate(
        self: "OpenAIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        # ask openai to confirm the item is correct
        counter.increment(CounterItem.AI_QUERY, item_config.name)
        prompt = self.get_prompt(listing, item_config, marketplace_config)
        res: AIResponse | None = AIResponse.from_cache(listing, item_config, marketplace_config)
        if res is not None:
            if self.logger:
                self.logger.debug(
                    f"""{hilight("[AI]", res.style)} {self.config.name} previously concluded {hilight(f"{res.conclusion} ({res.score}): {res.comment}", res.style)} for listing {hilight(listing.title)}."""
                )
            return res

        self.connect()

        retries = 0
        while retries < self.config.max_retries:
            self.connect()
            assert self.client is not None
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model or self.default_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can confirm if a user's search criteria matches the item he is interested in.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"""{hilight("[AI-Error]", "fail")} {self.config.name} failed to evaluate {hilight(listing.title)}: {e}"""
                    )
                retries += 1
                # try to initiate a connection
                self.client = None
                time.sleep(5)

        # check if the response is yes
        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Response]", "info")} {pretty_repr(response)}""")

        answer = response.choices[0].message.content or ""
        if (
            answer is None
            or not answer.strip()
            or re.search(r"Rating[^1-5]*[1-5]", answer, re.DOTALL) is None
        ):
            counter.increment(CounterItem.FAILED_AI_QUERY, item_config.name)
            raise ValueError(f"Empty or invalid response from {self.config.name}: {response}")

        lines = answer.split("\n")
        # if any of the lines contains "Rating: ", extract the rating from it.
        score: int = 1
        comment = ""
        rating_line = None
        for idx, line in enumerate(lines):
            matched = re.match(r".*Rating[^1-5]*([1-5])[:\s]*(.*)", line)
            if matched:
                score = int(matched.group(1))
                comment = matched.group(2).strip()
                rating_line = idx
                continue
            if rating_line is not None:
                # if the AI puts comment after Rating, we need to include them
                comment += " " + line
        # if the AI puts the rating at the end, let us try to use the line before the Rating line
        if len(comment.strip()) < 5 and rating_line is not None and rating_line > 0:
            comment = lines[rating_line - 1]

        # remove multiple spaces, take first 30 words
        comment = " ".join([x for x in comment.split() if x.strip()]).strip()
        res = AIResponse(name=self.config.name, score=score, comment=comment)
        res.to_cache(listing, item_config, marketplace_config)
        counter.increment(CounterItem.NEW_AI_QUERY, item_config.name)
        return res


class DeepSeekBackend(OpenAIBackend):
    default_model = "deepseek-chat"
    base_url = "https://api.deepseek.com"

    @classmethod
    def get_config(cls: Type["DeepSeekBackend"], **kwargs: Any) -> DeekSeekConfig:
        return DeekSeekConfig(**kwargs)


class GeminiBackend(OpenAIBackend):
    """Google Gemini via its OpenAI-compatible endpoint."""

    default_model = "gemini-2.5-flash"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    @classmethod
    def get_config(cls: Type["GeminiBackend"], **kwargs: Any) -> GeminiConfig:
        return GeminiConfig(**kwargs)


class OllamaBackend(OpenAIBackend):
    default_model = "deepseek-r1:14b"

    @classmethod
    def get_config(cls: Type["OllamaBackend"], **kwargs: Any) -> OllamaConfig:
        return OllamaConfig(**kwargs)


class AnthropicBackend(AIBackend):
    default_model = "claude-sonnet-4-20250514"

    @classmethod
    def get_config(cls: Type["AnthropicBackend"], **kwargs: Any) -> AnthropicConfig:
        return AnthropicConfig(**kwargs)

    def connect(self: "AnthropicBackend") -> None:
        if self.client is None:
            import anthropic  # type: ignore

            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
            if self.logger:
                self.logger.info(f"""{hilight("[AI]", "name")} {self.config.name} connected.""")

    def evaluate(
        self: "AnthropicBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        counter.increment(CounterItem.AI_QUERY, item_config.name)
        prompt = self.get_prompt(listing, item_config, marketplace_config)
        res: AIResponse | None = AIResponse.from_cache(listing, item_config, marketplace_config)
        if res is not None:
            if self.logger:
                self.logger.debug(
                    f"""{hilight("[AI]", res.style)} {self.config.name} previously concluded {hilight(f"{res.conclusion} ({res.score}): {res.comment}", res.style)} for listing {hilight(listing.title)}."""
                )
            return res

        self.connect()

        retries = 0
        while retries < self.config.max_retries:
            self.connect()
            assert self.client is not None
            try:
                response = self.client.messages.create(
                    model=self.config.model or self.default_model,
                    max_tokens=1024,
                    system="You are a helpful assistant that can confirm if a user's search criteria matches the item he is interested in.",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"""{hilight("[AI-Error]", "fail")} {self.config.name} failed to evaluate {hilight(listing.title)}: {e}"""
                    )
                retries += 1
                self.client = None
                time.sleep(5)

        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Response]", "info")} {pretty_repr(response)}""")

        answer = response.content[0].text if response.content else ""
        if (
            answer is None
            or not answer.strip()
            or re.search(r"Rating[^1-5]*[1-5]", answer, re.DOTALL) is None
        ):
            counter.increment(CounterItem.FAILED_AI_QUERY, item_config.name)
            raise ValueError(f"Empty or invalid response from {self.config.name}: {response}")

        lines = answer.split("\n")
        score: int = 1
        comment = ""
        rating_line = None
        for idx, line in enumerate(lines):
            matched = re.match(r".*Rating[^1-5]*([1-5])[:\s]*(.*)", line)
            if matched:
                score = int(matched.group(1))
                comment = matched.group(2).strip()
                rating_line = idx
                continue
            if rating_line is not None:
                comment += " " + line
        if len(comment.strip()) < 5 and rating_line is not None and rating_line > 0:
            comment = lines[rating_line - 1]

        comment = " ".join([x for x in comment.split() if x.strip()]).strip()
        res = AIResponse(name=self.config.name, score=score, comment=comment)
        res.to_cache(listing, item_config, marketplace_config)
        counter.increment(CounterItem.NEW_AI_QUERY, item_config.name)
        return res


# A sticker SKU, not a series phrase like "43 Class C350 Series".
# Five or more characters, at least two letters and two digits.
_MODEL_TOKEN = re.compile(r"\b[A-Za-z0-9-]{5,}\b")
_VISION_MODEL = "google/gemini-3.1-flash-lite"
_OPENROUTER_URL = "https://openrouter.ai/api/v1"
_VISION_SYSTEM = (
    "You extract printed model numbers from product photos. "
    "Reply with JSON only, no markdown. "
    "Keys: brand, model, size_in, image_index, confidence. "
    "Use null when a value is not printed on the photos. "
    "Do not invent a SKU suffix. A series name is not a sticker SKU."
)


def has_full_model_token(title: str, description: str) -> bool:
    """Return True when title or description already contains a sticker-style model."""
    for token in _MODEL_TOKEN.findall(f"{title} {description}"):
        letters = sum(character.isalpha() for character in token)
        digits = sum(character.isdigit() for character in token)
        if letters >= 2 and digits >= 2:
            return True
    return False


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a vision reply into brand, model, size_in, image_index, and confidence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {
        "brand": _nullable_str(data.get("brand")),
        "model": _nullable_str(data.get("model")),
        "size_in": _nullable_str(data.get("size_in")),
        "image_index": _nullable_int(data.get("image_index")),
        "confidence": _nullable_str(data.get("confidence")),
    }


def _nullable_str(value: Any) -> str:
    """Return a stripped string, or empty when the model sent null."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "null":
        return ""
    return text


def _nullable_int(value: Any) -> Optional[int]:
    """Return an int, or None when the model sent null or a non-number."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _apply_vision_payload(listing: Listing, payload: Dict[str, Any]) -> None:
    """Copy a cached or fresh vision payload onto the listing."""
    listing.brand = _nullable_str(payload.get("brand"))
    listing.model = _nullable_str(payload.get("model"))
    listing.size_in = _nullable_str(payload.get("size_in"))
    listing.image_index = _nullable_int(payload.get("image_index"))
    listing.confidence = _nullable_str(payload.get("confidence"))


def _image_bytes_digest(image_bytes: List[bytes]) -> str:
    """Hash image bytes so a photo change misses the cache and a repeat hits it."""
    digest = hashlib.sha256()
    for blob in image_bytes:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return digest.hexdigest()


def _vision_store(local_cache: Cache | None) -> Cache:
    """Return the cache that holds vision results."""
    return cache if local_cache is None else local_cache


def _image_parts(title: str, description: str, image_bytes: List[bytes]) -> List[Dict[str, Any]]:
    """Build OpenAI-style content parts. Images are bytes, not CDN URLs."""
    parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Title: {title}\nDescription: {description}\n"
                "Read the printed brand, model, and size in inches from the photos."
            ),
        }
    ]
    for blob in image_bytes:
        encoded = base64.b64encode(blob).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return parts


def _openrouter_complete(parts: List[Dict[str, Any]], api_key: str) -> str:
    """Send image parts to Gemini Flash Lite on OpenRouter and return the text."""
    client: Any = OpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_URL,
        default_headers={
            "X-Title": "AI Marketplace Monitor",
            "HTTP-Referer": "https://github.com/BoPeng/ai-marketplace-monitor",
        },
    )
    response: Any = client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {"role": "user", "content": parts},
        ],
        extra_body={"reasoning": {"effort": "minimal"}},
        stream=False,
    )
    return response.choices[0].message.content or ""


def read_model_from_photos(
    listing: Listing,
    image_bytes: List[bytes],
    logger: Logger | None = None,
    local_cache: Cache | None = None,
    complete: Callable[[List[Dict[str, Any]]], str] | None = None,
) -> None:
    """Fill listing model fields from photo bytes, using the image-bytes cache.

    Does not change the text deal-rating prompt. A missing key skips the call.
    ``complete`` is the HTTP call, injected by tests.
    """
    if not image_bytes:
        return
    store = _vision_store(local_cache)
    digest = _image_bytes_digest(image_bytes)
    cache_key = (CacheType.MODEL_VISION.value, digest)
    cached = store.get(cache_key)
    if isinstance(cached, dict):
        _apply_vision_payload(listing, cached)
        return

    caller = complete
    if caller is None:
        api_key = os.environ.get("OPENROUTER_KEY_AIMM")
        if not api_key:
            if logger:
                logger.debug("OPENROUTER_KEY_AIMM is unset. Skipping model photo read.")
            return

        def caller(parts: List[Dict[str, Any]]) -> str:
            return _openrouter_complete(parts, api_key)

    try:
        raw = caller(_image_parts(listing.title, listing.description, image_bytes))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if logger:
            logger.debug(f"{hilight('[AI]', 'fail')} Model photo read failed: {exc}")
        return

    payload = parse_model_json(raw)
    if payload is None:
        if logger:
            logger.debug(f"{hilight('[AI]', 'fail')} Model photo read was not JSON: {raw[:200]}")
        return
    store.set(cache_key, payload, tag=CacheType.MODEL_VISION.value)
    _apply_vision_payload(listing, payload)
