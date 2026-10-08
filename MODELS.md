# MODELS.md — when to use what

quick routing guide for vspye. pick the right mode per input, per use case, per audio condition. no background, no philosophy. match the modality to the right commands and let the tool pick the rest.

---

## per-modality dispatch

vspye auto-runs the right extractors per file type:

| input | extractors that run (default) |
|---|---|
| `image.jpg` | Florence-2 caption + OCR + pixel/palette + QR + entities |
| `video.mp4` | Florence-2 captions on N keyframes + OCR + pixel + audio extract+transcribe + QR on keyframes + entities (image + audio) |
| `audio.wav` | whisper transcript + entities (audio) |
| `folder/` | batch dispatch of the above across every supported file |

`--text-only` collapses any of those to the synthesized `context` block (the prose paragraph an LLM can consume directly).

---

## flag decision rules

| flag | when to add it |
|---|---|
| `--no-caption` | you're feeding image/video to a downstream LLM that already understands images — the caption is redundant. saves the 500MB Florence-2 download. |
| `--no-ocr` | the input is a photo or video without text. |
| `--no-pixel` | batch-pass on thousands of files; the palette + brightness isn't needed for the downstream task. |
| `--no-audio` | the video is a screen recording, gameplay, or any clip with no speech. saves whisper load. |
| `--no-qr` | the input is a photograph; no QR codes expected. saves ~50ms. (default: ON, ~zero cost) |
| `--no-entities` | the input has no text in OCR or audio. (default: ON, ~zero cost) |
| `--no-context` | you want the raw JSON, not the synthesized prose block. rare. |
| `--detect` | not exposed. vspye is a context reader, not a detection tool. for object labels, use Florence-2 `<OD>` task or a separate model. |
| `--quiet` / `-q` | always when piping. suppresses progress to stderr. |
| `--text-only` / `-t` | when piping to an LLM chat. prints just the prose context block, not the JSON. |

---

## Florence-2 task selection

Florence-2 supports several "task tokens" that change what it returns:

| task | output | when to use |
|---|---|---|
| `<MORE_DETAILED_CAPTION>` (current default) | 1-2 sentence prose description | most use cases. balanced. |
| `<CAPTION>` | 1 short sentence (~10 words) | when you want a one-liner for thumbnails or search indexing. |
| `<OD>` | bounding boxes + classes (Florence's own detection) | when you want object labels from Florence-2. edit `task = '<OD>'` in vspye.py to switch. |
| `<DENSE_REGION_CAPTION>` | one caption per detected region | layout-aware captioning for charts, slides, infographics. slow. |
| `<MORE_DETAILED_CAPTION>` + grounding | phrases → pixel boxes | when downstream needs to know "where in the image did Florence see X?" |

the current default in `caption_image()` is `<MORE_DETAILED_CAPTION>`. to change, edit `task = '<CAPTION>'` (or any other token) in `vspye.py`. no flag exposed; tracked at the model layer.

---

## whisper model selection

| model | size | accuracy on clear speech | accuracy on noisy | CPU speed (70s audio) |
|---|---|---|---|---|
| `tiny` | 39MB | ~85% | ~50% | ~3s |
| `base` (current default) | 75MB | ~95% | ~65% | ~7s |
| `small` | 244MB | ~96% | ~75% | ~14s |
| `medium` | 770MB | ~97% | ~80% | ~30s |
| `large-v3` | 1.5GB | ~98% | ~85% | ~60s (CPU) / ~7s (GPU) |

rules:
- clear voice, podcast, screen recording, voice memo: `base` (default).
- noisy / café / traffic / music bleed: swap to `large-v3`. edit `WhisperModel('base', ...)` to `WhisperModel('large-v3', ...)` in `_load_whisper()`. +3GB model, +2-3× wall time, +10-15% accuracy on noisy input.
- long audio (1h+) on CPU: stay on `base`. large-v3 is too slow on CPU for >30min content.
- lyrics: do not expect good output regardless of model. whisper is speech-trained. the right tool for lyrics is Demucs (vocal separation) → whisper, a different tool.

---

## when to skip layers

for a folder of 1000 mixed images, default settings run 6 extractors per file. that's slow. skip layers aggressively:

```bash
vspye.py ./screenshots/ --no-caption --no-pixel --no-qr      # OCR + entities only
vspye.py ./product-photos/ --no-ocr --no-audio               # captions + pixel + entities
vspye.py ./voice-memos/ --no-caption --no-pixel --no-qr --no-entities  # transcript only
```

the synthesized `context` block adapts — if a layer is skipped, that section is omitted from the prose.

---

## GPU auto-detection

`_gpu()` is called once per process. it returns `True` iff `torch.cuda.is_available()` is `True`. safe fallback is `False`.

when `True`, these layers run on GPU:

| layer | GPU benefit |
|---|---|
| Florence-2 captioning | 5-10× faster |
| whisper transcription | 3-5× faster (CTranslate2 CUDA) |

when `False` (or no torch), everything runs on CPU. no fallback needed; the device string passed to models is `'cpu'`.

CPU vs GPU wall time on a 70s video (insta2.mp4):

| stage | CPU | GPU (RTX 3060) |
|---|---|---|
| Florence-2 (4 keyframes) | ~8s | ~1.5s |
| whisper base (70s audio) | ~7s | ~2s |
| OCR + pixel + QR + entities | ~2s | ~2s (CPU-bound) |

total CPU: ~17s. total GPU: ~5.5s. ~3× end-to-end speedup on a CUDA box.

---

## audio model selection rule

| audio | model | reason |
|---|---|---|
| clear voice | base | default, fast, accurate enough |
| noisy / café / traffic | large-v3 | +10-15% accuracy, ~3GB cost, GPU strongly preferred |
| music + speech | large-v3 + VAD | VAD skips instrumental regions; large-v3 handles bleed better |
| lyrics | base or large (both will be inaccurate) | use Demucs → whisper instead, different tool |
| music-only | do not transcribe | features.mood is what you want, not transcript |

---

## false expectations

- vspye will **not** transcribe song lyrics accurately. don't try.
- vspye will **not** recognize handwriting. both OCR engines are printed-text only.
- vspye will **not** identify faces. out of scope.
- vspye will **not** run live streams. whole-file in, whole-file out.
- vspye will **not** call any API. nothing leaves the box.
- vspye will **not** load pretrained models you don't have. Florence-2 and whisper download on first use, then cache forever.

---

## quick recipes

```bash
# image → prose context for an LLM chat
vspye.py photo.jpg --text-only

# podcast → transcript only
vspye.py episode.mp3 --text-only

# video → full multimodal context
vspye.py clip.mp4 --text-only

# batch of mixed media
vspye.py ./media/ --quiet > out.json

# screenshot OCR only (no caption, no pixel)
vspye.py screen.png --no-caption --no-pixel --no-qr

# noisy audio with larger whisper (manual swap, see "whisper model selection")
# edit vspye.py: WhisperModel('base', ...) → WhisperModel('large-v3', ...)
vspye.py noisy_meeting.mp3 --no-caption --no-pixel --no-qr

# benchmark GPU vs CPU
time vspye.py clip.mp4 --quiet --text-only     # CPU
time vspye.py clip.mp4 --quiet --text-only     # GPU (same command, auto-detected)
```