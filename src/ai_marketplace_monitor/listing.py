from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Type

from diskcache import Cache  # type: ignore

from .utils import CacheType, cache, hash_dict


@dataclass
class Listing:
    marketplace: str
    name: str
    # unique identification
    id: str
    title: str
    image: str
    price: str
    post_url: str
    location: str
    seller: str
    condition: str
    description: str
    # Carousel URLs and sticker fields. Defaults keep old cache rows loadable.
    # Excluded from hash so a photo or a model read does not re-rate the deal.
    images: List[str] = field(default_factory=list)
    brand: str = ""
    model: str = ""
    size_in: str = ""
    image_index: Optional[int] = None
    confidence: str = ""

    @property
    def content(self: "Listing") -> Tuple[str, str, str]:
        return (self.title, self.description, self.price)

    @property
    def hash(self: "Listing") -> str:
        """Hash the deal identity, ignoring photos and extracted model fields."""
        # post_url query changes every search. Photos and sticker reads must not either.
        skipped = {"image", "images", "brand", "model", "size_in", "image_index", "confidence"}
        return hash_dict(
            {
                x: (y.split("?")[0] if x == "post_url" else y)
                for x, y in asdict(self).items()
                if x not in skipped
            }
        )

    @classmethod
    def from_cache(
        cls: Type["Listing"],
        post_url: str,
        local_cache: Cache | None = None,
    ) -> Optional["Listing"]:
        try:
            # details could be a different datatype, miss some key etc.
            # and we have recently changed to save Listing as a dictionary
            return cls(
                **(cache if local_cache is None else local_cache).get(
                    (CacheType.LISTING_DETAILS.value, post_url.split("?")[0])
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            return None

    def to_cache(
        self: "Listing",
        post_url: str,
        local_cache: Cache | None = None,
    ) -> None:
        (cache if local_cache is None else local_cache).set(
            (CacheType.LISTING_DETAILS.value, post_url.split("?")[0]),
            asdict(self),
            tag=CacheType.LISTING_DETAILS.value,
        )
