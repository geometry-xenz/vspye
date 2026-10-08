# vspye

> read images, video, and audio locally. emit prose context that any text-based model can consume.

---

## what it is

`vspye` is a single-file Python CLI. point it at a file (or folder) and it returns a JSON with prose captions, OCR, audio transcript, detected URLs/emails/phones, QR codes, and pixel stats — assembled into one `context` block ready to feed any LLM.

designed for software engineers and agentic builders who need to give a text-only model the actual content of an image / video / audio file. no GPU required, no API calls, no cloud accounts.

---

## why this exists

there are plenty of vision tools that output object labels — "person, bus, dog" — which is useless for an LLM that needs to *understand* what's happening. vspye outputs prose ("a man standing in front of a yellow bus holding a phone, the bus has Spanish signage on the side") because that's what text models can reason about.

this is a context reader, not a detector. no COCO classes. no bounding boxes from a detection model. semantic descriptions, OCR text, audio transcripts, and structured entities — that's what it produces.

---

## install

```bash
git clone <this-repo>
cd vspye

pip install onnxruntime opencv-python numpy

uv venv .venv
.venv/bin/python -m pip install torch transformers sentencepiece einops Pillow timm faster-whisper
```

runtime auto-detection:
- no venv → vspye auto-relaunches into `.venv/bin/python` for `--caption` or `--audio`
- no tesseract → falls back to rapidocr-onnxruntime (English only)
- no ffmpeg → `--audio` and video frame extraction fail with a clear error

optional:
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng    # Debian/Ubuntu
sudo pacman -S tesseract tesseract-data-eng        # Arch
brew install tesseract tesseract-lang              # macOS
choco install tesseract                            # Windows
```

GPU: nothing to install. `_gpu()` is called once per process; Florence-2 + faster-whisper auto-route to CUDA when `torch.cuda.is_available()` is True. no fallback logic needed.

---

## commands

| command | what runs |
|---|---|
| `vspye photo.jpg` | Florence-2 caption + OCR + pixel/palette + QR + entities |
| `vspye clip.mp4` | Florence-2 captions on N keyframes + OCR + pixel + audio transcript + QR on keyframes + entities |
| `vspye speech.wav` | whisper transcript + entities |
| `vspye ./folder/` | batch dispatch across every supported file |

flags:

| flag | what it skips |
|---|---|
| `--no-caption` | Florence-2 |
| `--no-ocr` | OCR |
| `--no-pixel` | pixel/palette analysis |
| `--no-audio` | audio extraction + transcription |
| `--no-qr` | QR detection |
| `--no-entities` | entity regex pass |
| `--no-context` | synthesized context block |
| `--quiet` / `-q` | progress to stderr |
| `--text-only` / `-t` | print only the `context` block |
| `--lang <code>` | OCR language (default: eng, auto with tesseract) |
| `--keyframes <n>` | video keyframes to caption (default 4) |

---

## supported formats

| type | extensions |
|---|---|
| image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.gif` |
| video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv` |
| audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.opus`, `.wma` |

---

## output schema

```json
{
  "tool": "vspye",
  "input": "photo.jpg",
  "type": "image" | "video" | "audio" | "batch",

  "caption": "...",
  "captions": [{"timestamp": 0.0, "caption": "..."}],

  "ocr": {
    "text": "full extracted text",
    "word_count": int,
    "language": "eng",
    "regions": [{"text", "bbox", "conf"}],
    "engine": "tesseract" | "rapidocr"
  },

  "audio": {
    "language", "language_probability", "duration_sec",
    "segment_count",
    "segments": [{"start", "end", "text"}],
    "text",
    "engine": "faster-whisper-base"
  },

  "pixel": {
    "width", "height",
    "mean_rgb", "std_rgb",
    "brightness", "is_grayscale", "edge_density",
    "palette": [{"hex", "pct"}]
  },

  "qrcodes": [{"decoded", "bbox"}],

  "entities": {
    "urls":   [{"value", "position": {"source", "offset"}}],
    "emails": [...],
    "phones": [...],
    "dates":  [...],
    "money":  [...]
  },

  "context": "VISUAL: ...\n\nTEXT IN IMAGE: at (x1,y1)-(x2,y2): \"...\"\n\nAPPEARANCE: ...\n\nQR CODES: ...\n\nEXTRACTED:\n  URLS (...): ...\n  EMAILS (...): ...\n\nSPEECH: \"...\""
}
```

the `context` field is the headline. sections appear only when content exists. entity positions carry `source` (`image` or `audio`) + character offset for downstream grounding.

---

## compute budget (CPU, single-threaded)

| input | cold cache | warm cache |
|---|---|---|
| image (caption + OCR + pixel + QR + entities) | 2-4s | 0.5-1s |
| video 60s, 4 keyframes + audio | 18-25s | 14-18s |
| audio 60s, whisper only | 7-10s | ~12s |

GPU bonus (CUDA-capable box, RTX 3060-class): Florence-2 5-10× faster, whisper 3-5× faster. ~3× end-to-end on a 60s video.

first-time setup downloads ~500MB Florence-2 + ~75MB whisper-base to `~/.cache/huggingface/`. cached forever.

---

## limitations (honest)

| input | what vspye does |
|---|---|
| clear voice, mild background | works (~95%) |
| noisy voice, café/traffic | degrades (~60-70%) |
| lyrics from a song | poor (~30-50%, hallucinates) |
| handwriting | not supported |
| faces | not identified |
| live streams / RTSP | not supported (whole-file only) |
| HEIC photos | not supported (convert first) |

for higher accuracy on noisy audio, swap whisper-base for `large-v3` in `_load_whisper()` (~3GB, ~3× slower on CPU, ~10% better accuracy on noisy input). for lyrics, use Demucs → whisper — a different tool.

---

## what it explicitly does NOT do

- **object detection.** no COCO labels. no `bus, person, dog` JSON arrays. if you need object labels, use Florence-2 `<OD>` task (`task = '<OD>'` in `caption_image()`) — same model, different output, gives you bboxes + classes.
- **face recognition.**
- **handwriting OCR.**
- **translation.** OCR returns text in source language.
- **stream processing.**
- **API calls.** nothing leaves the box.

---

## file layout

```
vspye/
├── vspye.py     single-file CLI (~640 LOC, one screen)
├── MODELS.md    when-to-use-what routing guide
└── README.md    this file
```

no `tests/`, no `tests.py`, no fixtures. the file is small enough that smoke-testing it on a real image + video + audio file is the entire regression suite. add new tests when the file gets bigger.

---

## license

MIT. copy, modify, ship, sell, no attribution required.