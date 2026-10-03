from __future__ import annotations


def device_info() -> dict:
    info = {
        "device": "cpu",
        "device_name": "CPU",
        "vram_total_mb": None,
        "vram_free_mb": None,
    }
    try:
        import torch
    except Exception:
        return info

    if not torch.cuda.is_available():
        return info

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    info.update(
        {
            "device": "cuda",
            "device_name": props.name,
            "vram_total_mb": int(total_bytes / (1024 * 1024)),
            "vram_free_mb": int(free_bytes / (1024 * 1024)),
        }
    )
    return info
