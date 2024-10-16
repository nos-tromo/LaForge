# LaForge

![LaForge-Logo](static/img/laforge_logo.png)

This project does OCR and machine translation to facilitate batch processing of image data.

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.


## Setup
Clone the repository and create a virtual environment:
```bash
git clone https://github.com/nos-tromo/LaForge.git
cd LaForge
pyenv install 3.11.6
pyenv local 3.11.6
python -m venv .venv
```
Set GPU-related environment variables before installing Python dependencies (see [pytorch.org](https://pytorch.org/) for troubleshooting):
```bash
# Cuda
echo 'export CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1' >> .venv/bin/activate
# Metal (Apple Silicon)
echo 'export CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1' >> .venv/bin/activate
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> .venv/bin/activate
```
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Download one of the [Gemma 2](https://blog.google/technology/developers/google-gemma-2/) models. The following quant 
sizes are suggestions depending on your machine's hardware. Comment out the model files that are not required:
```bash
directory=gguf
mkdir -p $directory/ && 
curl -L -o $directory/gemma-2-9b-it-Q8_0.gguf \
https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf && \
curl -L -o $directory/gemma-2-9b-it-Q4_K_M.gguf \
https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf && \
curl -L -o $directory/gemma-2-2b-it-Q8_0.gguf \
https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf
```

Running `LaForge` for the first time will download the `surya` models to `~/.cache/huggingface/hub`. After that, you can 
switch to offline mode:

```bash
echo 'export HF_HUB_OFFLINE=1' >> .venv/bin/activate
```

## Usage
Basic usage:
```bash
python laforge.py [path/to/file] [lang] [translate]
```
To detect multiple languages:
```bash
python laforge.py [path/to/data] [lang_1,lang_2...lang_n]
```
Batch processing:
```bash
python laforge.py [directory] [lang]
```
Add translation to OCR results:
```bash
python laforge.py [directory] [lang] [translate]
```
File output is stored under `output`.

## Important note
Multilingual detection is possible, but will likely result in significant accuracy loss. It is recommended to 
split your data into batches of the same languages before processing it.

## Examples
Single file processing:
```bash
python laforge.py data/image.png en
```
```bash
python laforge.py data/menu.jpg es,fr 
```
Batch processing:
```bash
python laforge.py fotos de
```
```bash
python laforge.py docs ch_sim
```
Translate results:
```bash
python laforge.py data it translate
```