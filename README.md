# LaForge

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

This project does OCR and machine translation to facilitate batch processing of image data.

## Setup
```bash
git clone https://github.com/nos-tromo/LaForge.git
cd LaForge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Running LaForge for the first time will download the required models to `~/.cache/huggingface/hub`. After that, you can 
switch to offline mode:

```bash
echo 'export HF_HUB_OFFLINE=1' >> .venv/bin/activate
```

## Usage
Basic usage:
```bash
python laforge.py [lang] [path/to/file] [translate]
```
To detect multiple languages:
```bash
python laforge.py [lang_1,lang_2...lang_n] [path/to/data]
```
Batch processing:
```bash
python laforge.py [lang] [directory]
```
Add translation to OCR results:
```bash
python laforge.py [lang] [directory] [translate]
```
File output is stored under `output`.

## Important note
Multilingual detection is possible, but will likely result in significant accuracy loss. It is recommended to 
split your data into batches of the same languages before processing it.

## Examples
Single file processing:
```bash
python laforge.py en data/image.png
```
```bash
python laforge.py es,fr data/menu.jpg 
```
Batch processing:
```bash
python laforge.py de fotos
```
```bash
python laforge.py ch_sim docs
```
Translate results:
```bash
python laforge.py it data translate
```