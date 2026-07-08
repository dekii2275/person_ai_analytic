# Environment Setup

## Recommendation

Use Conda on the local development machine.

```text
Environment manager: Conda
Environment name: orin_person
Python: 3.10
```

## Why Conda for this project?

- The developer already uses Conda for AI projects.
- Easy to isolate Python versions.
- Safer when multiple PyTorch/CUDA projects coexist.
- Easy to remove and recreate the environment.
- Later ONNX/TensorRT experiments can be isolated in another environment if necessary.

## Create environment

```bash
conda create -n orin_person python=3.10 -y
conda activate orin_person
python -m pip install --upgrade pip
```

## Verify local GPU

```bash
nvidia-smi
```

## Install PyTorch

Do not guess the CUDA command.

Use the official PyTorch installation selector for the current local machine.

After installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected:

```text
torch.cuda.is_available() == True
```

for GPU inference.

## Install milestone dependencies

```bash
pip install ultralytics opencv-python numpy
```

## Verify

```bash
python -c "import cv2, ultralytics, torch; print('OpenCV:', cv2.__version__); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Environment export

After the baseline works:

```bash
conda env export --from-history > environment.yml
pip freeze > requirements-lock.txt
```

Keep `requirements.txt` minimal and human-maintained.

## Important

Do not install yet:

```text
onnx
onnxruntime-gpu
tensorrt
deepstream
tracking libraries
face libraries
```

They are outside the current milestone.
