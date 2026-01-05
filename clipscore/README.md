# CLIPScore

CLIPScore is a Python framework for calculating a similarity score for query-image pairs using modular "adapters" to connect to different CLIP APIs or models (local or remote). The framework is service-agnostic and can easily be extended to support different CLIP backends or inference services.

## Features

- **Adapter-based architecture**: Plug in different CLIP APIs or models via simple adapters.
- **Service-agnostic**: Use CLIPScore with remote HTTP APIs (e.g., Hugging Face, Replicate, etc.) or implement your own local scoring.
- **Extensible**: Add your own adapter by subclassing `CLIPAdapter` and register it.
- **Auto-discovery**: List and select available adapters using the registry.
- **Failover/retry logic**: HTTP adapters come with robust error handling and retries.

## Usage

### Basic Example

```python
from clipscore.adapters import LocalCLIPAdapter

adapter = LocalCLIPAdapter(model="openai/clip-vit-base-patch32", device="cpu")
score = adapter.score(query="a photo of a fire truck", image_url="https://example.com/firetruck.jpg")
print("CLIPScore:", score)
```

### Creating Your Own Adapter

Implement a new adapter by subclassing `CLIPAdapter`:

```python
from clipscore.framework.base import CLIPAdapter, CLIPAdapterRegistry

class MyCLIPAdapter(CLIPAdapter):
    def score(self, query: str, image_url: str) -> float:
        # Your scoring logic
        return 0.0

    def get_name(self) -> str:
        return "my_clip_adapter"

# Register your adapter
CLIPAdapterRegistry.register("my_clip_adapter", MyCLIPAdapter)
```

Now it's available via the registry!

## Included Adapters

- **HTTPCLIPAdapter**: Basic HTTP REST API adapter (for any generic CLIP service).
- **LocalCLIPAdapter**: Built-in support for local CLIP models using Hugging Face transformers.

## Design

- **clipscore/framework/base.py** — Contains the abstract `CLIPAdapter` and the `CLIPAdapterRegistry`.
- **clipscore/adapters/clip.py** — Contains adapter implementations for HTTP CLIP APIs and local CLIP models using Hugging Face transformers.

Adapters encapsulate the logic for communicating with various CLIP endpoints and returning a numerical CLIPScore (typically between 0–1, or -1–1).

## Extending

To add a new CLIP backend:
1. Subclass `CLIPAdapter`, implement `score` and `get_name`.
2. Register your adapter with `CLIPAdapterRegistry`.
3. Use it via the registry in your code!

## Example: Registering and Using a Custom Adapter

```python
class YourAdapter(CLIPAdapter):
    def score(self, query, image_url):
        # Implement your integration
        return 42.0

    def get_name(self):
        return "your_adapter"

CLIPAdapterRegistry.register("your_adapter", YourAdapter)
adapter = CLIPAdapterRegistry.get("your_adapter")
print(adapter.score("cat", "https://..."))
```

## See Also

- [CLIP: Contrastive Language–Image Pretraining (OpenAI)](https://github.com/openai/CLIP)

