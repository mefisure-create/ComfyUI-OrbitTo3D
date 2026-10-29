import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
}

__submodules = ["samplers"]
__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):
    """Load the retained image-to-3D pipeline from a local/HF pipeline config."""
    import json
    import os

    is_local = os.path.exists(f"{path}/pipeline.json")
    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, "r") as f:
        config = json.load(f)

    name = config["name"]
    if name != "Trellis2ImageTo3DPipeline":
        raise ValueError(
            f"ComfyUI-OrbitTo3D retains only Trellis2ImageTo3DPipeline, got {name!r}"
        )
    return __getattr__(name).from_pretrained(path)
