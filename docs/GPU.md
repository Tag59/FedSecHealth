# GPU acceleration

The tabular experiments (v0.1) run in seconds to minutes on CPU; a GPU only
matters for the imaging milestone (v0.2). The code is device-agnostic:
`fl.device: auto` picks `cuda` if PyTorch sees a GPU. ROCm builds of PyTorch
expose AMD GPUs through the same `torch.cuda` API.

## AMD Radeon (ROCm) under WSL2

1. Update the Windows AMD Adrenalin driver to a version that supports WSL.
2. Check AMD's *ROCm on WSL* compatibility matrix for your GPU
   (RDNA3 cards such as the RX 7000 series are the target family, but support
   varies per model and driver version).
3. Install ROCm inside WSL following AMD's guide (`amdgpu-install --usecase=wsl,rocm --no-dkms`).
4. Point `uv` to the ROCm PyTorch wheels by replacing the index in `pyproject.toml`:

   ```toml
   [[tool.uv.index]]
   name = "pytorch-cpu"            # keep the name, change the URL
   url = "https://download.pytorch.org/whl/rocm6.4"   # match your ROCm version
   explicit = true
   ```

   then `uv sync` and verify:

   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

If your card is not supported under WSL, keep the CPU build: MedMNIST (28×28)
experiments remain tractable on a modern multi-core CPU.

## NVIDIA (CUDA)

Same as above with `https://download.pytorch.org/whl/cu128` (or the CUDA version
matching your driver).
