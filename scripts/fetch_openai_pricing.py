#!/usr/bin/env python3
"""
Fetch OpenAI pricing from official website and save to JSON.
This script parses the OpenAI pricing page and extracts model prices.
"""

import html as html_lib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


PRICING_URL = "https://developers.openai.com/api/docs/pricing"
OUTPUT_FILE = "github_pages/pricing.json"
API_FILE = "github_pages/api.json"
HISTORY_FILE = "github_pages/history.json"
SUPPORTED_TIERS = ("standard", "batch", "flex", "priority")
PREFERRED_TIER_ORDER = ("standard", "batch", "flex", "priority")
# The pricing page renamed "Priority" processing to "Fast mode"; the API accepts both.
TIER_ALIASES = {"fast": "priority", "fast mode": "priority"}


def fetch_html(url: str) -> str:
    """
    Fetch the pricing page HTML.

    The page is server-rendered (Astro), and every pricing table's full data,
    including rows collapsed behind "All models", is embedded in the
    <astro-island> props, so no JavaScript rendering is needed.
    """
    print(f"Fetching {url}...")

    request = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        html_content = response.read().decode("utf-8")

    print(f"Fetched {len(html_content)} characters")
    return html_content


def parse_price(text: str) -> float:
    """Extract price from text like '$2.50', '2.50', etc."""
    if not text:
        return 0.0

    # Remove $ and whitespace
    cleaned = re.sub(r'[\$\s,]', '', text)

    # Try to extract first number
    match = re.search(r'(\d+\.?\d*)', cleaned)
    if match:
        return float(match.group(1))
    return 0.0


def normalize_model_key(model_name: str) -> str:
    """Normalize model name for use as a JSON key."""
    return model_name.lower().replace(' ', '-').replace('·', '-')


def normalize_text(text: str) -> str:
    """Normalize text for heading/tier comparisons."""
    return ' '.join(text.strip().lower().split())


def normalize_tier(text: Optional[str]) -> Optional[str]:
    """Map a tier label from the page to a supported tier name, if it is one."""
    if not text:
        return None
    tier = normalize_text(text)
    tier = TIER_ALIASES.get(tier, tier)
    return tier if tier in SUPPORTED_TIERS else None


def infer_table_tier(table) -> str:
    """Infer pricing tier from the closest preceding heading-like element."""
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'strong', 'em']

    for element in table.find_all_previous(heading_tags, limit=100):
        tier = normalize_tier(element.get_text(" ", strip=True))
        if tier:
            return tier

    return "standard"


def classify_model(model_name: str) -> str:
    """Determine a model's category from its name."""
    model_name_lower = model_name.lower()

    if 'gpt-image' in model_name_lower or 'chatgpt-image' in model_name_lower:
        return 'image_generation_token'
    if any(x in model_name_lower for x in ['dall-e', 'dall·e']):
        return 'image_generation'
    if 'sora' in model_name_lower:
        return 'video_generation'
    if any(x in model_name_lower for x in ['whisper', 'transcribe', 'translate']):
        return 'audio_transcription'
    if 'tts' in model_name_lower:
        return 'text_to_speech'
    if 'embedding' in model_name_lower:
        return 'embeddings'
    if any(x in model_name_lower for x in ['o1', 'o3', 'o4']) and 'mini' not in model_name_lower:
        return 'reasoning'
    if any(x in model_name_lower for x in ['gpt-6', 'gpt-5', 'gpt-4', 'gpt-3.5', 'davinci', 'babbage']):
        return 'language_model'
    if 'computer-use' in model_name_lower:
        return 'computer_use'
    if 'storage' in model_name_lower:
        return 'storage'
    return 'other'


def build_tier_payload(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a tier-specific pricing payload from extracted model data."""
    return {
        key: value for key, value in model_data.items()
        if key not in ['model', 'timestamp', 'default_tier', 'available_tiers', 'tiers']
    }


TOP_LEVEL_FIELDS = [
    "pricing_type", "category", "input", "output", "cached_input", "cache_write", "long_context",
    "modalities", "price", "price_by_size", "price_per_minute", "image_pricing", "training", "training_unit",
]


def sync_default_tier_fields(entry: Dict[str, Any]) -> None:
    """Expose a stable top-level view using the preferred default tier."""
    tier_data = entry.get("tiers", {})
    if not tier_data:
        return

    default_tier = next((tier for tier in PREFERRED_TIER_ORDER if tier in tier_data), None)
    if not default_tier:
        default_tier = next(iter(tier_data))

    entry["default_tier"] = default_tier
    entry["available_tiers"] = [tier for tier in PREFERRED_TIER_ORDER if tier in tier_data]

    for field in TOP_LEVEL_FIELDS:
        if field in tier_data[default_tier]:
            entry[field] = tier_data[default_tier][field]
        elif field in entry:
            del entry[field]


def upsert_tiered_model(
    pricing: Dict[str, Any],
    model_name: str,
    model_data: Dict[str, Any],
    tier: str,
) -> None:
    """Create or update a model entry with tier-specific pricing."""
    model_key = normalize_model_key(model_name)
    tier_payload = build_tier_payload(model_data)

    if model_key not in pricing:
        pricing[model_key] = {
            "model": model_name,
            "timestamp": model_data["timestamp"],
            "tiers": {},
        }

    entry = pricing[model_key]
    entry["tiers"][tier] = tier_payload

    if len(model_name) > len(entry["model"]):
        entry["model"] = model_name

    sync_default_tier_fields(entry)


def parse_image_resolution_table(table, headers: list, pricing: Dict[str, Any], tier: str) -> None:
    """Parse image resolution pricing tables (e.g., 1024x1024, quality-based)."""
    print(f"    Parsing image resolution table for {tier} tier...")

    # Extract resolution headers (e.g., "1024 x 1024", "1024 x 1536")
    resolution_indices = {}
    for idx, header in enumerate(headers):
        if 'x' in header and any(char.isdigit() for char in header):
            # Normalize resolution format: "1024 x 1024" -> "1024x1024"
            resolution = re.sub(r'\s*x\s*', 'x', header)
            resolution_indices[idx] = resolution

    print(f"    Resolution columns: {resolution_indices}")

    # Get quality column index (if exists)
    quality_idx = None
    for idx, header in enumerate(headers):
        if 'quality' in header:
            quality_idx = idx
            break

    rows = table.find_all('tr')[1:]  # Skip header

    current_model = None
    for row_idx, row in enumerate(rows, 1):
        cells = row.find_all(['td', 'th'])
        if len(cells) < 2:
            continue

        # First cell might be model name or quality
        first_cell = cells[0].get_text(strip=True)

        # Check if this is a quality label
        if first_cell.lower() in ['low', 'medium', 'high', 'standard', 'hd']:
            quality = first_cell.lower()
        # Check if this is a model name
        elif len(first_cell) > 3 and any(c.isalpha() for c in first_cell):
            # Skip header-like names
            if first_cell.lower() in ['model', 'quality']:
                continue
            # Valid model name
            current_model = first_cell
            print(f"    Row {row_idx}: Model = {current_model}")
            quality = 'standard'  # default quality for new model
        else:
            quality = 'standard'  # default

        # Try to find quality in dedicated column if exists
        if quality_idx and quality_idx < len(cells):
            quality_text = cells[quality_idx].get_text(strip=True).lower()
            # Only use if it's a valid quality label
            if quality_text in ['low', 'medium', 'high', 'standard', 'hd']:
                quality = quality_text

        # Skip if no model identified
        if not current_model:
            continue

        # Extract prices for each resolution
        resolution_prices = {}
        for idx, resolution in resolution_indices.items():
            if idx < len(cells):
                price = parse_price(cells[idx].get_text(strip=True))
                if price > 0:
                    resolution_prices[resolution] = price

        if not resolution_prices:
            continue

        model_key = normalize_model_key(current_model)

        if model_key not in pricing:
            pricing[model_key] = {
                "model": current_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tiers": {},
            }

        entry = pricing[model_key]
        tier_entry = entry["tiers"].setdefault(
            tier,
            {
                "pricing_type": "per_image_resolution",
                "category": "image_generation_token",
                "image_pricing": {},
            },
        )
        tier_entry["image_pricing"][quality] = resolution_prices

        if len(current_model) > len(entry["model"]):
            entry["model"] = current_model

        sync_default_tier_fields(entry)

        print(f"      {quality}: {resolution_prices}")


def parse_pricing_html(html: str) -> Dict[str, Any]:
    """Parse OpenAI pricing HTML and extract model prices.

    Prefers the structured data embedded in the page's Astro component props,
    falling back to scraping rendered <table> elements.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')

    pricing = parse_pricing_islands(soup)
    if pricing:
        return pricing

    print("No pricing component data found, falling back to table parsing")
    return parse_pricing_tables(soup)


def decode_astro_value(value: Any) -> Any:
    """Decode Astro's serialized props format ([type, value] pairs)."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], int):
        kind, inner = value
        if kind == 0:
            return decode_astro_value(inner) if isinstance(inner, dict) else inner
        if kind == 1:
            return [decode_astro_value(item) for item in inner]
        return inner
    if isinstance(value, dict):
        return {key: decode_astro_value(item) for key, item in value.items()}
    return value


def cell_text(value: Any) -> str:
    """Convert a decoded pricing cell (plain value, HTML, or tooltip heading) to text."""
    from bs4 import BeautifulSoup

    if isinstance(value, dict):
        if "__pricingHtml" in value:
            fragment = BeautifulSoup(value["__pricingHtml"], "html.parser")
            for small in fragment.find_all("small"):
                if normalize_text(small.get_text()) == "legacy":
                    small.decompose()
            return " ".join(fragment.get_text(" ", strip=True).split())
        if "__pricingTooltipHeading" in value:
            return cell_text(value["__pricingTooltipHeading"].get("label", ""))
        return ""
    if value is None:
        return ""
    return html_lib.unescape(str(value)).strip()


def cell_price(value: Any) -> Optional[float]:
    """Convert a pricing cell to a float price, or None when it has no price."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = cell_text(value)
    if normalize_text(text) == "free":
        return 0.0
    if not re.search(r'\d', text):
        return None
    return parse_price(text)


def clean_model_name(name: str) -> str:
    """Strip annotations like '(<272K context length)' from a model name."""
    return re.sub(r'\s*\([^)]*context[^)]*\)\s*$', '', name).strip()


TOKEN_HEADER_FIELDS = {
    "input": "input",
    "cached input": "cached_input",
    "cache writes": "cache_write",
    "output": "output",
    "output / cost": "output",
}


def token_fields(headers: List[str], values: List[Any]) -> Dict[str, float]:
    """Map token price columns (by header) to payload fields."""
    fields: Dict[str, float] = {}
    long_context: Dict[str, float] = {}

    for header, value in zip(headers, values):
        target = fields
        if header.startswith("short context "):
            header = header[len("short context "):]
        elif header.startswith("long context "):
            header = header[len("long context "):]
            target = long_context

        field = TOKEN_HEADER_FIELDS.get(header)
        if isinstance(value, str) and "/" in value:
            continue  # A non-token unit price, e.g. "$15.00 / 1M characters"
        price = cell_price(value)
        if field and price is not None:
            target[field] = price

    if long_context:
        fields["long_context"] = long_context
    return fields


def make_model_data(model_name: str, category: Optional[str] = None, **fields: Any) -> Dict[str, Any]:
    """Build a model_data dict in the shape upsert_tiered_model expects."""
    return {
        "model": model_name,
        "category": category or classify_model(model_name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


def token_model_data(model_name: str, fields: Dict[str, Any], category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build per-1M-token model data, or None if no token prices were found."""
    if not any(key in fields for key in ("input", "output", "cached_input")):
        return None
    return make_model_data(model_name, category, pricing_type="per_1m_tokens", **fields)


def unit_price_model_data(model_name: str, text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build model data from a unit price like '$0.006 / minute' or '$15.00 / 1M characters'."""
    price = cell_price(text)
    if price is None:
        return None

    lowered = text.lower()
    if "minute" in lowered:
        pricing_type = "per_minute"
    elif "second" in lowered:
        pricing_type = "per_second"
    elif "1m characters" in lowered:
        pricing_type = "per_1m_chars"
    elif "1k characters" in lowered or "1k chars" in lowered:
        pricing_type = "per_1k_chars"
    else:
        return None

    return make_model_data(model_name, category, pricing_type=pricing_type, price=price)


def parse_modality_group(model_name: str, headers: List[str], rows: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """Parse a model whose rows are split by modality (Audio / Text / Image)."""
    price_headers = headers[2:]
    modalities: Dict[str, Dict[str, float]] = {}
    unit_price_data = None

    for row in rows:
        modality = normalize_text(cell_text(row[0]))
        fields = token_fields(price_headers, row[1:])
        if fields:
            modalities[modality] = fields
        elif unit_price_data is None:
            for value in row[1:]:
                unit_price_data = unit_price_model_data(model_name, cell_text(value))
                if unit_price_data:
                    break

    if not modalities:
        return unit_price_data

    # Top-level token prices come from the first modality listed (Audio for
    # realtime/audio models, Image for image models), matching earlier output,
    # with gaps filled from later modalities (e.g. TTS: text input, audio output).
    primary: Dict[str, float] = {}
    for fields in reversed(list(modalities.values())):
        primary.update(fields)
    return token_model_data(model_name, {**primary, "modalities": modalities})


def parse_video_group(model_name: str, headers: List[str], rows: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """Parse a video model with per-size, per-second pricing."""
    price_idx = next((i for i, h in enumerate(headers[1:]) if "second" in h), None)
    if price_idx is None:
        return None

    price_by_size: Dict[str, float] = {}
    for row in rows:
        price = cell_price(row[price_idx]) if price_idx < len(row) else None
        if price is not None:
            price_by_size[cell_text(row[0])] = price

    if not price_by_size:
        return None

    return make_model_data(
        model_name,
        pricing_type="per_second",
        price=next(iter(price_by_size.values())),
        price_by_size=price_by_size,
    )


def parse_use_case_group(model_name: str, headers: List[str], rows: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """Parse a transcription-style model: token prices and/or an estimated per-minute cost."""
    row = rows[0]
    values = dict(zip(headers[1:], row))
    fields = token_fields(headers[2:], row[1:])
    estimate = unit_price_model_data(model_name, cell_text(values.get("estimated cost")))

    data = token_model_data(model_name, fields)
    if data and estimate:
        data["price_per_minute"] = estimate["price"]
    return data or estimate


def parse_pricing_component(props: Dict[str, Any], tier: str, pricing: Dict[str, Any]) -> None:
    """Parse one pricing component's props into the pricing dict."""
    # Flagship text-token tables: rows are [model, input, cached input, (cache writes,) output]
    if "tier" in props and "rows" in props and "headings" not in props:
        tier = normalize_tier(props["tier"]) or tier
        for row in props["rows"]:
            model_name = clean_model_name(cell_text(row[0]))
            columns = ["input", "cached input", "cache writes", "output"]
            if len(row) == 4:
                columns = ["input", "cached input", "output"]
            data = token_model_data(model_name, token_fields(columns, row[1:]))
            if data:
                upsert_tiered_model(pricing, model_name, data, tier)
        return

    headers = [normalize_text(cell_text(h)) for h in props.get("headings", [])]
    if not headers:
        return

    entries = []  # (model_name, model_data)

    if "rows" in props:
        category = "fine_tuning" if "training" in headers else None
        for row in props["rows"]:
            model_name = clean_model_name(cell_text(row[0]))
            values = dict(zip(headers[1:], row[1:]))
            data = token_model_data(model_name, token_fields(headers[1:], row[1:]), category)
            if data is None:
                for header, value in values.items():
                    if "minute" in header and cell_price(value) is not None:
                        data = make_model_data(model_name, category, pricing_type="per_minute", price=cell_price(value))
            if data and category == "fine_tuning":
                training = cell_text(values.get("training"))
                data["training"] = cell_price(values.get("training"))
                data["training_unit"] = "per_hour" if "hour" in training.lower() else "per_1m_tokens"
            entries.append((model_name, data))

    for group in props.get("groups", []):
        group_name = cell_text(group.get("model"))
        rows = group.get("rows", [])
        if not rows:
            continue

        if headers[0] == "tool":
            continue  # Tool pricing isn't per-model
        if headers[0] == "category":
            for row in rows:
                model_name = clean_model_name(cell_text(row[0]))
                entries.append((model_name, token_model_data(model_name, token_fields(headers[2:], row[1:]))))
            continue

        model_name = clean_model_name(group_name)
        if len(headers) > 1 and headers[1] == "modality":
            data = parse_modality_group(model_name, headers, rows)
        elif any("second" in h for h in headers):
            data = parse_video_group(model_name, headers, rows)
        elif len(headers) > 1 and headers[1] == "use case":
            data = parse_use_case_group(model_name, headers, rows)
        else:
            data = token_model_data(model_name, token_fields(headers[1:], rows[0]))
        entries.append((model_name, data))

    for model_name, data in entries:
        if not data:
            continue
        # Fine-tuning tables reuse some base model names (e.g. gpt-3.5-turbo);
        # don't let fine-tuned inference prices overwrite base model prices.
        existing = pricing.get(normalize_model_key(model_name))
        if data["category"] == "fine_tuning" and existing and existing.get("category") != "fine_tuning":
            continue
        upsert_tiered_model(pricing, model_name, data, tier)


def parse_pricing_islands(soup) -> Dict[str, Any]:
    """Extract pricing from the structured props of the page's Astro pricing components."""
    pricing: Dict[str, Any] = {}
    islands = [
        island for island in soup.find_all('astro-island')
        if 'pricing' in (island.get('component-url') or '') and island.get('props')
    ]
    print(f"Found {len(islands)} pricing components")

    for island in islands:
        try:
            raw_props = json.loads(island['props'])
        except json.JSONDecodeError:
            continue
        props = {key: decode_astro_value(value) for key, value in raw_props.items()}

        pane = island.find_parent(attrs={'data-content-switcher-pane': True})
        tier = normalize_tier(pane.get('data-value')) if pane else None
        tier = tier or "standard"

        heading = island.find_previous(['h2', 'h3'])
        print(f"  {island.get('component-export')} ({heading.get_text(strip=True) if heading else '?'}, {tier})")
        parse_pricing_component(props, tier, pricing)

    return pricing


def parse_pricing_tables(soup) -> Dict[str, Any]:
    """Parse rendered pricing <table> elements (legacy page layout)."""
    pricing = {}

    # Find all tables on the page
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")

    for table_idx, table in enumerate(tables):
        print(f"\nProcessing table {table_idx + 1}...")
        tier = infer_table_tier(table)
        print(f"  Inferred tier: {tier}")

        # Get headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        else:
            # Try first row as headers
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True).lower() for th in first_row.find_all(['th', 'td'])]

        print(f"  Headers: {headers}")

        # Check if this is an image resolution pricing table
        if any('x' in h and any(char.isdigit() for char in h) for h in headers):
            print(f"  Detected image resolution pricing table")
            parse_image_resolution_table(table, headers, pricing, tier)
            continue

        # Skip if no relevant headers
        if not any(keyword in ' '.join(headers) for keyword in ['model', 'input', 'output', 'price']):
            print(f"  Skipping table (no pricing headers)")
            continue

        # Get all rows
        rows = table.find_all('tr')

        for row_idx, row in enumerate(rows[1:], 1):  # Skip header row
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue

            # First cell is usually model name
            model_name = cells[0].get_text(strip=True)

            # Skip invalid model names
            if not model_name or len(model_name) < 3:
                continue
            if model_name.lower() in ['model', 'tier', '', 'models']:
                continue
            if not any(c.isalnum() for c in model_name):
                continue

            print(f"  Row {row_idx}: {model_name}")

            # Extract prices based on headers
            model_data = {
                "model": model_name,
                "pricing_type": "unknown",
                "category": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Try to find input/output prices (for language models)
            for idx, header in enumerate(headers):
                if idx >= len(cells):
                    break

                cell_text = cells[idx].get_text(strip=True)
                price = parse_price(cell_text)

                # Language models (tokens)
                if 'input' in header and 'cached' not in header and price > 0:
                    model_data['input'] = price
                    model_data['pricing_type'] = 'per_1m_tokens'
                elif 'output' in header and price > 0:
                    model_data['output'] = price
                    model_data['pricing_type'] = 'per_1m_tokens'
                elif 'cached input' in header and price > 0:
                    model_data['cached_input'] = price
                # Audio models
                elif 'minute' in header and price > 0:
                    model_data['price'] = price
                    model_data['pricing_type'] = 'per_minute'
                # Video models
                elif 'second' in header and price > 0:
                    model_data['price'] = price
                    model_data['pricing_type'] = 'per_second'
                # Generic price - determine by model name
                elif 'price' in header and price > 0:
                    model_data['price'] = price

            # Determine pricing type and category from model name if not already set
            model_name_lower = model_name.lower()

            # Determine pricing type if still unknown
            if model_data['pricing_type'] == 'unknown' and 'price' in model_data:
                # GPT models (language)
                if any(x in model_name_lower for x in ['gpt', 'o1', 'o3', 'o4']):
                    model_data['pricing_type'] = 'per_1m_tokens'
                # Image generation models
                elif any(x in model_name_lower for x in ['dall-e', 'dall·e']):
                    model_data['pricing_type'] = 'per_image'
                # Video models
                elif 'sora' in model_name_lower:
                    model_data['pricing_type'] = 'per_second'
                # Audio transcription
                elif 'whisper' in model_name_lower:
                    model_data['pricing_type'] = 'per_minute'
                # Text-to-speech
                elif 'tts' in model_name_lower:
                    model_data['pricing_type'] = 'per_1k_chars'
                # Embeddings
                elif 'embedding' in model_name_lower:
                    model_data['pricing_type'] = 'per_1m_tokens'

            model_data['category'] = classify_model(model_name)

            # Only add if has meaningful pricing data
            has_pricing = any(k in model_data for k in ['input', 'output', 'price', 'cached_input'])

            if has_pricing:
                upsert_tiered_model(pricing, model_name, model_data, tier)

                print(f"    Extracted: {model_data}")

    return pricing


def create_api_json(pricing: Dict[str, Any]) -> Dict[str, Any]:
    """Create simplified API JSON."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return {
        "models": pricing,
        "timestamp": timestamp,
        "last_updated": timestamp,
        "models_count": len(pricing),
        "source": "openai_official_pricing_page",
    }


def update_history(pricing: Dict[str, Any]) -> None:
    """Update pricing history JSON file."""
    history = []
    
    # Load existing history
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        pass
    
    # Create today's entry
    today = datetime.now(timezone.utc).date().isoformat()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    today_entry = {
        "date": today,
        "timestamp": timestamp,
        "models": pricing,
        "models_count": len(pricing),
    }
    
    # Update or append
    updated = False
    for i, entry in enumerate(history):
        if entry.get('date') == today:
            history[i] = today_entry
            updated = True
            break
    
    if not updated:
        history.append(today_entry)
    
    # Keep only last 90 days
    history = sorted(history, key=lambda x: x['date'], reverse=True)[:90]
    
    # Save
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\nHistory updated: {len(history)} days")


def main():
    """Main function."""
    try:
        # Fetch HTML
        html = fetch_html(PRICING_URL)
        
        # Parse pricing
        pricing = parse_pricing_html(html)
        
        if not pricing:
            print("\nWARNING: No pricing data extracted!")
            sys.exit(1)
        
        print(f"\n\nExtracted {len(pricing)} models")
        
        # Save full pricing data
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(pricing, f, indent=2, ensure_ascii=False)
        print(f"Saved to {OUTPUT_FILE}")
        
        # Save API JSON
        api_data = create_api_json(pricing)
        with open(API_FILE, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {API_FILE}")
        
        # Update history
        update_history(pricing)
        
        print("\n✓ Success!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
