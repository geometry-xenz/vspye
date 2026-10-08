#!/usr/bin/env python3
import argparse
import json
import os.path
import re
import sys
import tempfile
import subprocess



EXT_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
EXT_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
EXT_AUDIO = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma'}

_quiet = False


def ext_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_IMAGES:
        return 'image'
    if ext in EXT_VIDEOS:
        return 'video'
    if ext in EXT_AUDIO:
        return 'audio'
    return None


def _venv_python():
    if os.name == 'nt':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '.venv', 'Scripts', 'python.exe')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '.venv', 'bin', 'python')


def _venv_deps_ok():
    try:
        import torch  # noqa
        import transformers  # noqa
        from PIL import Image  # noqa
        return True
    except ImportError:
        return False


def _ensure_caption_deps():
    if _venv_deps_ok():
        return
    venv_py = _venv_python()
    if not os.path.exists(venv_py):
        sys.stderr.write(
            'caption + audio require a venv with ML deps.\n'
            f'  expected: {venv_py}\n'
            '  run from project root:\n'
            '    uv venv .venv\n'
            '    .venv/bin/python -m pip install torch==2.14.1 transformers==4.46.3 \\\n'
            '        sentencepiece einops Pillow timm faster-whisper\n'
        )
        sys.exit(1)
    sys.stderr.write(f'relaunching with {venv_py}\n')
    os.execv(venv_py, [venv_py] + sys.argv)


def _parse_lang(lang_arg):
    if not lang_arg:
        return 'eng'
    return lang_arg


def _ocr_tesseract(path, lang):
    r = subprocess.run(
        ['tesseract', path, '-', '-l', lang],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return None
    text = r.stdout.strip()
    words = text.split()
    return {
        'text': text,
        'word_count': len(words),
        'language': lang,
        'is_text_heavy': len(words) >= 50,
        'engine': 'tesseract',
    }


def _ocr_rapidocr(path):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    eng = RapidOCR(params={'lang_rec': 'english'})
    try:
        result = eng(path)
    except Exception as e:
        sys.stderr.write(f'ocr failed: {e}\n')
        return None
    if not result or not result[0]:
        return {'text': '', 'word_count': 0, 'language': 'eng',
                'is_text_heavy': False, 'engine': 'rapidocr'}
    regions = []
    words = []
    for bbox, text, conf in result[0]:
        if not text:
            continue
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]
        regions.append({'text': text, 'bbox': [int(x1), int(y1), int(x3), int(y3)],
                        'conf': round(float(conf), 3)})
        words.extend(text.split())
    return {
        'text': ' '.join(words),
        'word_count': len(words),
        'language': 'eng',
        'is_text_heavy': len(words) >= 50,
        'regions': regions,
        'engine': 'rapidocr',
    }


def extract_ocr(path, lang):
    import shutil
    if shutil.which('tesseract'):
        return _ocr_tesseract(path, _parse_lang(lang))
    return _ocr_rapidocr(path)


def extract_pixel(path):
    import cv2
    import numpy as np
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, 256))
    arr = small.astype(np.float32)
    mean_bgr = arr.reshape(-1, 3).mean(axis=0)
    std_bgr = arr.reshape(-1, 3).std(axis=0)
    brightness = float(mean_bgr.mean() / 255.0)
    is_gray = bool(std_bgr.max() < 10)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge_density = float(cv2.Sobel(gray, cv2.CV_32F, 1, 1).var() / 10000.0)
    arr2 = small.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        arr2, 5, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=5)
    total = int(counts.sum())
    palette = []
    for i in range(5):
        b, g, r = (int(v) for v in centers[i])
        palette.append({
            'hex': f'#{r:02x}{g:02x}{b:02x}',
            'pct': round(counts[i] / total, 3),
        })
    palette.sort(key=lambda c: -c['pct'])
    return {
        'width': w,
        'height': h,
        'mean_rgb': [round(float(v), 1) for v in mean_bgr[::-1]],
        'std_rgb': [round(float(v), 1) for v in std_bgr[::-1]],
        'brightness': round(brightness, 3),
        'is_grayscale': is_gray,
        'edge_density': round(edge_density, 3),
        'palette': palette,
    }


_caption_model = None
_caption_processor = None


def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _device():
    return 'cuda' if _gpu() else 'cpu'


def _load_caption():
    global _caption_model, _caption_processor
    if _caption_model is not None:
        return
    from transformers import AutoProcessor, AutoModelForCausalLM
    if not _quiet:
        sys.stderr.write(f'loading florence-2-base-ft on {_device()} (one-time, ~500MB)...\n')
    _caption_processor = AutoProcessor.from_pretrained(
        'microsoft/Florence-2-base-ft', trust_remote_code=True,
    )
    _caption_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-base-ft', trust_remote_code=True,
    ).to(_device()).eval()


def caption_image(path):
    _ensure_caption_deps()
    import torch
    from PIL import Image
    _load_caption()
    img = Image.open(path).convert('RGB')
    w, h = img.size
    task = '<MORE_DETAILED_CAPTION>'
    inputs = _caption_processor(text=task, images=img, return_tensors='pt')
    with torch.no_grad():
        ids = _caption_model.generate(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    raw = _caption_processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = _caption_processor.post_process_generation(
        raw, task=task, image_size=(w, h),
    )
    return parsed[task].strip()


def caption_video_keyframes(path, n=4):
    _ensure_caption_deps()
    import torch
    from PIL import Image
    _load_caption()
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = min(n, total)
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        img = PILImage.fromarray(rgb)
        w, h = img.size
        task = '<MORE_DETAILED_CAPTION>'
        inputs = _caption_processor(text=task, images=img, return_tensors='pt')
        with torch.no_grad():
            ids = _caption_model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        raw = _caption_processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = _caption_processor.post_process_generation(
            raw, task=task, image_size=(w, h),
        )
        out.append({'timestamp': round(idx / max(cap.get(cv2.CAP_PROP_FPS), 1), 2),
                    'caption': parsed[task].strip()})
    cap.release()
    return out


_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return
    from faster_whisper import WhisperModel
    if not _quiet:
        sys.stderr.write(f'loading whisper-base on {_device()} (one-time, ~75MB)...\n')
    _whisper_model = WhisperModel('base', device=_device(), compute_type='int8')


def transcribe(path, start_offset=0.0):
    _ensure_caption_deps()
    _load_whisper()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
        wav = t.name
    try:
        r = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y', '-i', path,
             '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            return None
        segs, info = _whisper_model.transcribe(wav, beam_size=5, vad_filter=True)
        seg_list = list(segs)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    return {
        'language': info.language,
        'language_probability': round(float(info.language_probability), 3),
        'duration_sec': round(float(info.duration), 2),
        'segment_count': len(seg_list),
        'segments': [
            {'start': round(s.start + start_offset, 2),
             'end': round(s.end + start_offset, 2),
             'text': s.text.strip()}
            for s in seg_list
        ],
        'text': ' '.join(s.text.strip() for s in seg_list),
        'engine': 'faster-whisper-base',
    }


def _audio_duration(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0 and r.stdout.strip():
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    return 0.0


def _chunk_audio(path, chunk_min):
    if chunk_min <= 0:
        return [(path, 0.0)]
    duration = _audio_duration(path)
    if not duration or duration <= chunk_min * 60:
        return [(path, 0.0)]
    wavs = []
    start = 0.0
    while start < duration:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        r = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y',
             '-ss', str(start), '-i', path,
             '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
             '-t', str(chunk_min * 60), tmp.name],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            try: os.unlink(tmp.name)
            except OSError: pass
            return wavs
        wavs.append((tmp.name, start))
        start += chunk_min * 60
    return wavs


def extract_audio(path, args=None):
    import shutil
    if not shutil.which('ffmpeg'):
        return None
    if args is None or args.chunk <= 0 or args.workers <= 1:
        return transcribe(path)
    chunks = _chunk_audio(path, args.chunk)
    n = len(chunks)
    if n <= 1:
        return transcribe(path)
    if not _quiet:
        est_mb = n * 600
        sys.stderr.write(
            f'chunked: {n} chunks × ~{est_mb} MB peak RAM, {args.workers} workers\n')
    from concurrent.futures import ProcessPoolExecutor
    results = [None] * n
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {i: pool.submit(transcribe, wav, off)
                for i, (wav, off) in enumerate(chunks)}
        for i, fut in futs.items():
            results[i] = fut.result()
    for wav, _ in chunks:
        try: os.unlink(wav)
        except OSError: pass
    results = [r for r in results if r]
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    merged = {
        'language': results[0]['language'],
        'language_probability': results[0]['language_probability'],
        'duration_sec': sum(r['duration_sec'] for r in results),
        'segment_count': sum(r['segment_count'] for r in results),
        'segments': sorted(
            (s for r in results for s in r['segments']),
            key=lambda s: s['start']),
        'text': ' '.join(r['text'] for r in results if r['text']),
        'engine': 'faster-whisper-base',
    }
    return merged


_PATTERNS = {
    'urls':   r'https?://[^\s<>"\'\u2019]+',
    'emails': r'[\w.+-]+@[\w-]+\.[\w.-]+',
    'phones': r'(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}',
    'dates':  r'\d{4}-\d{2}-\d{2}',
    'money':  r'[\$€£¥]\s?[\d,]+(?:\.\d{2})?',
}


def extract_entities(text, origin=None):
    out = {k: [] for k in _PATTERNS}
    if not text:
        return out
    for kind, pat in _PATTERNS.items():
        seen = set()
        for m in re.finditer(pat, text):
            v = m.group(0).rstrip('.,;:')
            if not v or v in seen:
                continue
            seen.add(v)
            out[kind].append({
                'value': v,
                'position': {'source': origin, 'offset': m.start()},
            })
            if len(out[kind]) >= 20:
                break
    return out


def _qr_from_arr(arr):
    import cv2
    d = cv2.QRCodeDetector()
    val, points, _ = d.detectAndDecode(arr)
    if points is None or not val:
        return []
    out = []
    vals = val if isinstance(val, (list, tuple)) else [val]
    for i, v in enumerate(vals):
        if not v:
            continue
        pts = points[i] if i < len(points) else points[0]
        out.append({
            'decoded': v,
            'bbox': [int(min(pts[:, 0])), int(min(pts[:, 1])),
                     int(max(pts[:, 0])), int(max(pts[:, 1]))],
        })
    return out


def detect_qr(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return []
    return _qr_from_arr(img)


def detect_qr_video(path, n=4):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = min(max(n, 1), total)
    out = []
    for i in range(n):
        idx = int(i * (total - 1) / max(n - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        for q in _qr_from_arr(fr):
            q['frame_ts'] = round(idx / fps, 2)
            out.append(q)
    cap.release()
    return out


def synthesize_context(caption, ocr, pixel, audio, detections=None, qrcodes=None, entities=None):
    parts = []
    if caption:
        if isinstance(caption, list):
            parts.append('VISUAL: ' + ' | '.join(
                f"t={c['timestamp']}s: {c['caption']}" for c in caption))
        else:
            parts.append(f'VISUAL: {caption}')
    if ocr and ocr.get('regions'):
        lines = []
        for r in ocr['regions']:
            if not r.get('text'):
                continue
            x1, y1, x2, y2 = r['bbox']
            lines.append(f'  - at ({x1},{y1})-({x2},{y2}): "{r["text"]}"')
        if lines:
            parts.append('TEXT IN IMAGE:\n' + '\n'.join(lines))
    elif ocr and ocr.get('text'):
        parts.append(f'TEXT IN IMAGE: "{ocr["text"]}"')
    if qrcodes:
        lines = [f'  - at ({q["bbox"][0]},{q["bbox"][1]})-({q["bbox"][2]},{q["bbox"][1]}): "{q["decoded"]}"'
                 for q in qrcodes]
        parts.append('QR CODES:\n' + '\n'.join(lines))
    if entities:
        lines = []
        for kind, items in entities.items():
            if items:
                src = items[0].get('position', {}).get('source', '?')
                vals = ', '.join(e['value'] for e in items)
                lines.append(f'  {kind.upper()} ({src}): {vals}')
        if lines:
            parts.append('EXTRACTED:\n' + '\n'.join(lines))
    if pixel:
        top = pixel.get('palette', [{}])[0]
        parts.append(
            f"APPEARANCE: {pixel['width']}x{pixel['height']}, "
            f"brightness {pixel['brightness']:.2f}, "
            f"dominant {top.get('hex', '?')}"
        )
    if audio and audio.get('text'):
        t = audio['text']
        parts.append(f'SPEECH: "{t[:500]}{"..." if len(t) > 500 else ""}"')
    elif audio and audio.get('language'):
        parts.append(
            f"AUDIO: {audio.get('duration_sec', 0)}s, language={audio['language']}, "
            f"{audio.get('segment_count', 0)} segments"
        )
    return '\n\n'.join(parts) if parts else ''


def _audio_meta(path):
    import shutil
    if not shutil.which('ffmpeg'):
        return None
    r = subprocess.run(
        ['ffmpeg', '-i', path],
        capture_output=True, text=True, timeout=30,
    )
    return r.stderr if r.returncode == 0 else None


def _merge_entities(a, b):
    out = {k: list(a.get(k, [])) for k in _PATTERNS}
    for k in _PATTERNS:
        seen = {e['value'] for e in out[k]}
        for e in b.get(k, []):
            if e['value'] not in seen:
                out[k].append(e)
                seen.add(e['value'])
            if len(out[k]) >= 20:
                break
    return out


def process_file(path, args):
    kind = ext_type(path)
    if kind is None:
        return None
    payload = {
        'tool': 'vspye',
        'input': path,
        'type': kind,
    }
    qrcodes = []
    entities = {k: [] for k in _PATTERNS}
    if kind == 'image':
        if not args.no_caption:
            try:
                payload['caption'] = caption_image(path)
            except Exception as e:
                sys.stderr.write(f'caption failed: {e}\n')
                payload['caption'] = None
        if not args.no_ocr:
            payload['ocr'] = extract_ocr(path, args.lang)
        if not args.no_pixel:
            payload['pixel'] = extract_pixel(path)
        if not args.no_qr:
            try:
                qrcodes = detect_qr(path)
            except Exception as e:
                sys.stderr.write(f'qr failed: {e}\n')
        if not args.no_entities and (payload.get('ocr') or {}).get('text'):
            entities = _merge_entities(
                entities, extract_entities(payload['ocr']['text'], 'image'))
    elif kind == 'video':
        if not args.no_caption:
            try:
                payload['captions'] = caption_video_keyframes(path, n=args.keyframes)
            except Exception as e:
                sys.stderr.write(f'caption failed: {e}\n')
                payload['captions'] = []
        if not args.no_ocr:
            payload['ocr'] = extract_ocr(path, args.lang)
        if not args.no_pixel:
            payload['pixel'] = extract_pixel(path)
        if not args.no_audio:
            payload['audio'] = extract_audio(path, args)
        if not args.no_qr:
            try:
                qrcodes = detect_qr_video(path, args.keyframes)
            except Exception as e:
                sys.stderr.write(f'qr failed: {e}\n')
        if not args.no_entities:
            if (payload.get('ocr') or {}).get('text'):
                entities = _merge_entities(
                    entities, extract_entities(payload['ocr']['text'], 'image'))
            if payload.get('audio', {}).get('text'):
                entities = _merge_entities(
                    entities, extract_entities(payload['audio']['text'], 'audio'))
    elif kind == 'audio':
        if not args.no_audio:
            payload['audio'] = extract_audio(path, args)
        if not args.no_entities and payload.get('audio', {}).get('text'):
            entities = extract_entities(payload['audio']['text'], 'audio')
    if qrcodes:
        payload['qrcodes'] = qrcodes
    if any(entities.values()):
        payload['entities'] = entities
    if not args.no_context:
            caption_for_ctx = payload.get('caption') or payload.get('captions')
            payload['context'] = synthesize_context(
                caption_for_ctx,
                payload.get('ocr'),
                payload.get('pixel'),
                payload.get('audio'),
                qrcodes=qrcodes,
                entities=entities,
            )
    return payload


def build_parser():
    p = argparse.ArgumentParser(
        prog='vspye',
        description='Read images, video, and audio. Output prose context for AI agents.',
    )
    p.add_argument('input', help='Image, video, audio file, or folder')
    p.add_argument('--no-caption', action='store_true', help='Skip Florence-2 caption')
    p.add_argument('--no-ocr', action='store_true', help='Skip OCR')
    p.add_argument('--no-pixel', action='store_true', help='Skip pixel analysis')
    p.add_argument('--no-audio', action='store_true', help='Skip audio transcription')
    p.add_argument('--no-qr', action='store_true', help='Skip QR detection')
    p.add_argument('--no-entities', action='store_true', help='Skip entity regex')
    p.add_argument('--no-context', action='store_true', help='Skip synthesized context block')
    p.add_argument('--lang', default='eng', help='OCR language (default: eng, auto with tesseract)')
    p.add_argument('--keyframes', type=int, default=4, help='Video keyframes to caption (default 4)')
    p.add_argument('--chunk', type=float, default=0, help='Chunk long audio into N-minute pieces for parallel whisper (0 = off)')
    p.add_argument('--workers', type=int, default=1, help='Parallel worker processes for chunked whisper (default 1 = sequential)')
    p.add_argument('--quiet', '-q', action='store_true', help='Suppress progress output')
    p.add_argument('--text-only', '-t', action='store_true', help='Print just the context block')
    return p


def main() -> int:
    global _quiet
    args = build_parser().parse_args()
    _quiet = args.quiet

    if not os.path.exists(args.input):
        sys.stderr.write(f'not found: {args.input}\n')
        return 2

    paths = []
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for f in files:
                full = os.path.join(root, f)
                if ext_type(full) is not None:
                    paths.append(full)
        paths.sort()
    else:
        paths.append(args.input)

    if not paths:
        sys.stderr.write(f'no supported files in {args.input}\n')
        return 2

    results = []
    for p in paths:
        if not _quiet and len(paths) > 1:
            sys.stderr.write(f'processing: {p}\n')
        try:
            r = process_file(p, args)
        except Exception as e:
            sys.stderr.write(f'{p}: failed: {e}\n')
            continue
        if r is None:
            continue
        results.append(r)

    if not results:
        sys.stderr.write('no results\n')
        return 1

    output = results[0] if len(results) == 1 else {
        'tool': 'vspye',
        'input': args.input,
        'type': 'batch',
        'count': len(results),
        'results': results,
    }

    if args.text_only:
        ctx = output.get('context') or ''
        if ctx:
            print(ctx)
        elif 'results' in output:
            for r in output['results']:
                c = r.get('context') or ''
                if c:
                    print(f"=== {r['input']} ===\n{c}\n")
    else:
        print(json.dumps(output, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
