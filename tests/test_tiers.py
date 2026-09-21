"""Regression tests for tier-aware pricing support."""

import html
import importlib.util
import json
from pathlib import Path

from openai_pricing_api import PricingCalculator
from openai_pricing_api.pricing import PricingProvider


def load_scraper_module():
    """Load the scraper script as a module for direct unit testing."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_openai_pricing.py"
    spec = importlib.util.spec_from_file_location("fetch_openai_pricing", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cache(cache_file: Path, models: dict) -> None:
    """Write a valid provider cache file for isolated tests."""
    cache_file.write_text(
        json.dumps(
            {
                "timestamp": "2099-01-01T00:00:00",
                "models": models,
            }
        ),
        encoding="utf-8",
    )


def test_parse_pricing_html_preserves_multiple_tiers_and_defaults_to_standard():
    """Repeated models should keep separate tier prices instead of being overwritten."""
    scraper = load_scraper_module()
    html = """
    <html>
      <body>
        <div>Standard</div>
        <table>
          <thead>
            <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
          </thead>
          <tbody>
            <tr><td>gpt-5</td><td>$1.25</td><td>$0.125</td><td>$10.00</td></tr>
            <tr><td>gpt-5-chat-latest</td><td>$1.25</td><td>$0.125</td><td>$10.00</td></tr>
          </tbody>
        </table>
        <div>Priority</div>
        <table>
          <thead>
            <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
          </thead>
          <tbody>
            <tr><td>gpt-5</td><td>$2.50</td><td>$0.25</td><td>$20.00</td></tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    pricing = scraper.parse_pricing_html(html)

    assert pricing["gpt-5"]["default_tier"] == "standard"
    assert pricing["gpt-5"]["available_tiers"] == ["standard", "priority"]
    assert pricing["gpt-5"]["input"] == 1.25
    assert pricing["gpt-5"]["output"] == 10.0
    assert pricing["gpt-5"]["tiers"]["standard"]["input"] == 1.25
    assert pricing["gpt-5"]["tiers"]["priority"]["input"] == 2.5
    assert pricing["gpt-5-chat-latest"]["tiers"]["standard"]["output"] == 10.0


def test_pricing_provider_resolves_requested_tier_from_cache(tmp_path):
    """Provider should expose tier-specific views from cached tiered model data."""
    cache_file = tmp_path / "pricing_cache.json"
    write_cache(
        cache_file,
        {
            "gpt-5": {
                "model": "gpt-5",
                "default_tier": "standard",
                "available_tiers": ["standard", "batch"],
                "tiers": {
                    "standard": {
                        "pricing_type": "per_1m_tokens",
                        "category": "language_model",
                        "input": 1.25,
                        "cached_input": 0.125,
                        "output": 10.0,
                    },
                    "batch": {
                        "pricing_type": "per_1m_tokens",
                        "category": "language_model",
                        "input": 0.625,
                        "cached_input": 0.0625,
                        "output": 5.0,
                    },
                },
                "input": 1.25,
                "cached_input": 0.125,
                "output": 10.0,
                "pricing_type": "per_1m_tokens",
                "category": "language_model",
            }
        },
    )

    provider = PricingProvider(cache_file=cache_file)
    default_pricing = provider.get_model_pricing("gpt-5")
    batch_pricing = provider.get_model_pricing("gpt-5", tier="batch")

    assert default_pricing is not None
    assert default_pricing.selected_tier == "standard"
    assert default_pricing.available_tiers == ["standard", "batch"]
    assert default_pricing.input_price == 1.25

    assert batch_pricing is not None
    assert batch_pricing.selected_tier == "batch"
    assert batch_pricing.input_price == 0.625
    assert provider.get_model_pricing("gpt-5", tier="priority") is None


def test_calculator_uses_requested_token_tier(tmp_path):
    """Token cost calculations should respect the selected pricing tier."""
    cache_file = tmp_path / "pricing_cache.json"
    write_cache(
        cache_file,
        {
            "gpt-5": {
                "model": "gpt-5",
                "default_tier": "standard",
                "available_tiers": ["standard", "batch"],
                "tiers": {
                    "standard": {
                        "pricing_type": "per_1m_tokens",
                        "category": "language_model",
                        "input": 1.25,
                        "cached_input": 0.125,
                        "output": 10.0,
                    },
                    "batch": {
                        "pricing_type": "per_1m_tokens",
                        "category": "language_model",
                        "input": 0.625,
                        "cached_input": 0.0625,
                        "output": 5.0,
                    },
                },
                "input": 1.25,
                "cached_input": 0.125,
                "output": 10.0,
                "pricing_type": "per_1m_tokens",
                "category": "language_model",
            }
        },
    )

    calculator = PricingCalculator(cache_file=cache_file)

    standard_cost = calculator.calculate_token_cost("gpt-5", input_tokens=1_000_000, output_tokens=0)
    batch_cost = calculator.calculate_token_cost(
        "gpt-5",
        input_tokens=1_000_000,
        output_tokens=0,
        tier="batch",
    )

    assert standard_cost == 1.25
    assert batch_cost == 0.625


def test_calculator_uses_requested_image_tier(tmp_path):
    """Image generation calculations should resolve prices from the requested tier."""
    cache_file = tmp_path / "pricing_cache.json"
    write_cache(
        cache_file,
        {
            "gpt-image-1": {
                "model": "gpt-image-1",
                "default_tier": "standard",
                "available_tiers": ["standard", "batch"],
                "tiers": {
                    "standard": {
                        "pricing_type": "per_image_resolution",
                        "category": "image_generation_token",
                        "image_pricing": {
                            "standard": {
                                "1024x1024": 0.04,
                            }
                        },
                    },
                    "batch": {
                        "pricing_type": "per_image_resolution",
                        "category": "image_generation_token",
                        "image_pricing": {
                            "standard": {
                                "1024x1024": 0.02,
                            }
                        },
                    },
                },
                "pricing_type": "per_image_resolution",
                "category": "image_generation_token",
                "image_pricing": {
                    "standard": {
                        "1024x1024": 0.04,
                    }
                },
            }
        },
    )

    calculator = PricingCalculator(cache_file=cache_file)

    assert calculator.calculate_image_cost("gpt-image-1", count=2) == 0.08
    assert calculator.calculate_image_cost("gpt-image-1", count=2, tier="batch") == 0.04


def astro_island(props: dict, tier: str | None = None) -> str:
    """Render a pricing <astro-island> the way the docs page serializes its props."""

    def encode(value):
        if isinstance(value, list):
            return [1, [encode(item) for item in value]]
        if isinstance(value, dict):
            return [0, {key: encode(item) for key, item in value.items()}]
        return [0, value]

    serialized = json.dumps({key: encode(value) for key, value in props.items()})
    island = (
        '<astro-island component-url="/_astro/pricing.abc.js" '
        f"props='{html.escape(serialized, quote=True)}'></astro-island>"
    )
    if tier:
        return f'<div data-content-switcher-pane="true" data-value="{tier}">{island}</div>'
    return island


def test_parse_pricing_html_reads_astro_component_props():
    """Current page layout: parse structured props, including collapsed rows and fast-mode tier."""
    scraper = load_scraper_module()
    page = "<html><body>" + "".join([
        astro_island({"tier": "standard", "rows": [
            ["gpt-6-astra", 10, 1, 12.5, 50],
            ["gpt-5.5 (<272K context length)", 5, 0.5, "-", 30],
            ["gpt-5-pro", 15, None, 120],
        ]}, "standard"),
        astro_island({"tier": "fast", "rows": [["gpt-6-astra", 20, 2, 25, 100]]}, "fast"),
        astro_island({
            "headings": ["Model", "Modality", "Input", "Cached input", "Output / cost"],
            "groups": [
                {"model": "gpt-realtime", "rows": [["Audio", 32, 0.4, 64], ["Text", 4, 0.4, 16]]},
                {"model": "tts-1", "rows": [["Text", "$15.00 / 1M characters", "-", "-"]]},
            ],
        }),
        astro_island({
            "headings": ["Model", "Size", "Portrait", "Landscape", "Price per second"],
            "groups": [{"model": "sora-2-pro", "rows": [
                ["720p", "720x1280", "1280x720", "$0.30"],
                ["1080p", "1080x1920", "1920x1080", "$0.70"],
            ]}],
        }, "batch"),
        astro_island({
            "headings": ["Category", "Model", "Input", "Cached input", "Output"],
            "groups": [{"model": "Embedding", "rows": [["text-embedding-3-small", 0.02, "-", "-"]]}],
        }, "standard"),
        astro_island({
            "headings": ["Model", "Training", "Input", "Cached input", "Output"],
            "rows": [
                [{"__pricingHtml": "o4-mini-2025-04-16<br /><small>with data sharing</small>"},
                 "$100.00 / hour", 2, 0.5, 8],
                [{"__pricingHtml": "gpt-5-pro<br /><small>Legacy</small>"}, 8, 3, None, 6],
            ],
        }, "standard"),
    ]) + "</body></html>"

    pricing = scraper.parse_pricing_html(page)

    astra = pricing["gpt-6-astra"]
    assert astra["available_tiers"] == ["standard", "priority"]
    assert astra["input"] == 10.0
    assert astra["cache_write"] == 12.5
    assert astra["tiers"]["priority"]["output"] == 100.0

    assert pricing["gpt-5.5"]["cached_input"] == 0.5
    assert "cache_write" not in pricing["gpt-5.5"]
    assert pricing["gpt-5-pro"]["output"] == 120.0
    assert "cached_input" not in pricing["gpt-5-pro"]

    assert pricing["gpt-realtime"]["input"] == 32.0
    assert pricing["gpt-realtime"]["modalities"]["text"]["output"] == 16.0
    assert pricing["tts-1"]["pricing_type"] == "per_1m_chars"
    assert pricing["tts-1"]["price"] == 15.0

    assert pricing["sora-2-pro"]["default_tier"] == "batch"
    assert pricing["sora-2-pro"]["price"] == 0.3
    assert pricing["sora-2-pro"]["price_by_size"]["1080p"] == 0.7

    assert pricing["text-embedding-3-small"]["input"] == 0.02
    assert "category" not in pricing and "embedding" not in pricing

    finetune = pricing["o4-mini-2025-04-16-with-data-sharing"]
    assert finetune["category"] == "fine_tuning"
    assert finetune["training"] == 100.0
    assert finetune["training_unit"] == "per_hour"
    # A fine-tuning row must not overwrite the base model's prices
    assert pricing["gpt-5-pro"]["category"] == "language_model"
    assert pricing["gpt-5-pro"]["input"] == 15.0
