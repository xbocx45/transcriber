"""
Сервер расшифровки речи в текст (на базе Whisper, работает без интернета и без оплаты за минуты).

Что умеет:
  * POST /v1/transcribe            расшифровать файл (голосовое, звонок, видео), свой формат ответа
  * POST /v1/audio/transcriptions  то же в формате OpenAI: подходит программам, которые умеют работать с OpenAI Whisper
  * GET  /v1/jobs/{id}             результат долгой расшифровки
  * GET  /v1/stats                 сколько минут расшифровано, по программам
  * GET  /health                   работает ли сервер
  * GET  /                         страница для ручной загрузки файла

Доступ только по ключу: заголовок "Authorization: Bearer КЛЮЧ" (или "X-Api-Key: КЛЮЧ").
Настройки берутся из переменных окружения (файл /etc/transcriber.env).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

VERSION = "1.0"

# ---------- настройки ----------
MODEL = os.environ.get("MODEL", "large-v3-turbo")
COMPUTE = os.environ.get("COMPUTE_TYPE", "int8")
THREADS = int(os.environ.get("CPU_THREADS", "0") or 0)
LANGUAGE = os.environ.get("LANGUAGE", "ru")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/var/lib/transcriber"))
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
WAIT_SECONDS = int(os.environ.get("WAIT_SECONDS", "900"))
JOB_TTL_DAYS = int(os.environ.get("JOB_TTL_DAYS", "7"))
FAKE_MODEL = os.environ.get("FAKE_MODEL") == "1"  # только для проверки без модели


def load_tokens() -> dict[str, str]:
    """TOKENS="bot:abc123,crm:def456" -> {"abc123": "bot", "def456": "crm"}. Имя нужно только для статистики."""
    out = {}
    for i, item in enumerate(filter(None, (x.strip() for x in os.environ.get("TOKENS", "").split(",")))):
        name, _, tok = item.rpartition(":")
        out[tok] = name or f"ключ {i + 1}"
    return out


TOKENS = load_tokens()

log = logging.getLogger("transcriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

for sub in ("jobs", "uploads", "models"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

# ---------- модель ----------
_model = None
_model_error = ""
_model_ready = threading.Event()


def load_model():
    global _model, _model_error
    if FAKE_MODEL:
        _model_ready.set()
        return
    try:
        from faster_whisper import WhisperModel

        log.info("Загружаю модель %s (%s)...", MODEL, COMPUTE)
        _model = WhisperModel(MODEL, device="cpu", compute_type=COMPUTE, cpu_threads=THREADS,
                              download_root=str(DATA_DIR / "models"))
        log.info("Модель готова")
    except Exception as e:  # noqa: BLE001
        _model_error = f"{type(e).__name__}: {e}"
        log.exception("Модель не загрузилась")
    finally:
        _model_ready.set()


def decode(path: str, split_stereo: bool = False):
    from faster_whisper import decode_audio

    return decode_audio(path, sampling_rate=16000, split_stereo=split_stereo)


def channel_count(path: str) -> int:
    try:
        import av

        with av.open(path) as c:
            s = next((s for s in c.streams if s.type == "audio"), None)
            return int(s.codec_context.channels) if s else 0
    except Exception:  # noqa: BLE001
        return 0


def run_whisper(audio, language: str, prompt: str):
    if FAKE_MODEL:
        dur = len(audio) / 16000
        return [{"start": 0.0, "end": round(dur, 2), "text": f"тестовая расшифровка {dur:.0f} сек"}], dur, language
    segs, info = _model.transcribe(
        audio, language=language or None, beam_size=5, vad_filter=True,
        condition_on_previous_text=False, initial_prompt=prompt or None,
    )
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs if s.text.strip()]
    return out, float(info.duration), info.language


def transcribe_file(path: str, opts: dict) -> dict:
    """Расшифровка с учетом режима: один голос или звонок с двумя дорожками."""
    language = opts.get("language") or LANGUAGE
    prompt = opts.get("prompt") or ""
    mode = opts.get("speakers") or "auto"
    names = [n.strip() for n in (opts.get("names") or "Собеседник 1,Собеседник 2").split(",")][:2]
    while len(names) < 2:
        names.append(f"Собеседник {len(names) + 1}")

    stereo = False
    if mode in ("auto", "stereo") and channel_count(path) == 2:
        left, right = decode(path, split_stereo=True)
        if mode == "stereo":
            stereo = True
        else:
            import numpy as np

            diff = float(np.mean(np.abs(left - right)))
            level = float(np.mean(np.abs(left)) + np.mean(np.abs(right))) / 2 + 1e-9
            stereo = diff / level > 0.3  # дорожки заметно разные: это запись звонка, а не просто стерео
    t0 = time.time()
    if stereo:
        segs = []
        duration = 0.0
        lang = language
        for name, audio in ((names[0], left), (names[1], right)):
            part, duration, lang = run_whisper(audio, language, prompt)
            segs += [dict(s, speaker=name) for s in part]
        segs.sort(key=lambda s: s["start"])
        lines = []
        for s in segs:
            if lines and lines[-1][0] == s["speaker"]:
                lines[-1][1].append(s["text"])
            else:
                lines.append((s["speaker"], [s["text"]]))
        dialog = "\n".join(f"{who}: {' '.join(t)}" for who, t in lines)
        return {"text": dialog, "dialog": dialog, "speakers": "stereo", "language": lang,
                "duration": round(duration, 1), "processing_seconds": round(time.time() - t0, 1), "segments": segs}
    audio = decode(path)
    segs, duration, lang = run_whisper(audio, language, prompt)
    return {"text": " ".join(s["text"] for s in segs).strip(), "speakers": "mono", "language": lang,
            "duration": round(duration, 1), "processing_seconds": round(time.time() - t0, 1), "segments": segs}


# ---------- очередь ----------
jobs: dict[str, dict] = {}
events: dict[str, threading.Event] = {}
work: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


def job_path(jid: str) -> Path:
    return DATA_DIR / "jobs" / f"{jid}.json"


def save_job(job: dict):
    tmp = job_path(job["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    tmp.replace(job_path(job["id"]))


def add_stats(client: str, seconds: float):
    f = DATA_DIR / "stats.json"
    with lock:
        try:
            st = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            st = {}
        month = time.strftime("%Y-%m")
        st.setdefault(month, {}).setdefault(client, {"files": 0, "minutes": 0.0})
        st[month][client]["files"] += 1
        st[month][client]["minutes"] = round(st[month][client]["minutes"] + seconds / 60, 1)
        f.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def worker():
    load_model()
    while True:
        jid = work.get()
        job = jobs.get(jid)
        if not job:
            continue
        job["status"] = "running"
        job["started"] = time.time()
        save_job(job)
        try:
            if _model is None and not FAKE_MODEL:
                raise RuntimeError("Модель не загружена: " + _model_error)
            job["result"] = transcribe_file(job["file"], job["options"])
            job["status"] = "done"
            add_stats(job["client"], job["result"].get("duration", 0))
            log.info("Готово %s: %s сек звука за %s сек (%s)", jid, job["result"]["duration"],
                     job["result"]["processing_seconds"], job["client"])
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            msg = f"{e}"
            if "Invalid data" in msg or "does not contain any stream" in msg:
                msg = "Файл не похож на аудио или видео"
            job["error"] = msg
            log.exception("Ошибка расшифровки %s", jid)
        finally:
            job["finished"] = time.time()
            try:
                os.remove(job["file"])
            except OSError:
                pass
            save_job(job)
            ev = events.pop(jid, None)
            if ev:
                ev.set()


def restore_jobs():
    """После перезапуска: готовые результаты остаются доступны, незаконченные ставятся в очередь заново."""
    limit = time.time() - JOB_TTL_DAYS * 86400
    for f in sorted((DATA_DIR / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            job = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if job.get("created", 0) < limit:
            f.unlink(missing_ok=True)
            continue
        jobs[job["id"]] = job
        if job["status"] in ("queued", "running"):
            if os.path.exists(job["file"]):
                job["status"] = "queued"
                work.put(job["id"])
            else:
                job["status"] = "error"
                job["error"] = "Файл потерялся при перезапуске сервера, отправьте его еще раз"
                save_job(job)


def cleanup_loop():
    while True:
        time.sleep(3600)
        limit = time.time() - JOB_TTL_DAYS * 86400
        for jid in [j for j, v in list(jobs.items()) if v.get("created", 0) < limit and v["status"] in ("done", "error")]:
            jobs.pop(jid, None)
            job_path(jid).unlink(missing_ok=True)


restore_jobs()
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=cleanup_loop, daemon=True).start()

# ---------- веб ----------
app = FastAPI(title="Расшифровка речи", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None)

_fails: dict[str, list[float]] = {}


def client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    return xff.split(",")[-1].strip() if xff else (req.client.host if req.client else "?")


def auth(req: Request) -> str | None:
    """Возвращает имя программы по ключу или None. После 20 ошибок за 10 минут адрес блокируется."""
    ip = client_ip(req)
    now = time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < 600]
    if len(recent) >= 20:
        return "__blocked__"
    h = req.headers.get("authorization", "")
    tok = h[7:].strip() if h.lower().startswith("bearer ") else req.headers.get("x-api-key", "").strip()
    for known, name in TOKENS.items():
        if tok and hmac.compare_digest(tok, known):
            return name
    recent.append(now)
    _fails[ip] = recent
    log.warning("Неверный ключ доступа с адреса %s", ip)
    return None


def deny(who):
    if who == "__blocked__":
        return JSONResponse({"error": "Слишком много неверных ключей. Подождите 10 минут."}, status_code=429)
    return JSONResponse({"error": "Нужен ключ доступа: заголовок Authorization: Bearer КЛЮЧ"}, status_code=401)


@app.middleware("http")
async def headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


def save_upload(file: UploadFile, jid: str) -> str:
    ext = Path(file.filename or "audio").suffix.lower()[:8] or ".bin"
    if not ext[1:].isalnum():
        ext = ".bin"
    dst = DATA_DIR / "uploads" / f"{jid}{ext}"
    size = 0
    with open(dst, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                out.close()
                dst.unlink(missing_ok=True)
                raise ValueError(f"Файл больше {MAX_MB} МБ")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise ValueError("Пустой файл")
    return str(dst)


def new_job(file: UploadFile, client: str, opts: dict) -> dict:
    jid = uuid.uuid4().hex
    path = save_upload(file, jid)
    job = {"id": jid, "status": "queued", "created": time.time(), "client": client, "options": opts,
           "file": path, "filename": (file.filename or "")[:200]}
    jobs[jid] = job
    events[jid] = threading.Event()
    save_job(job)
    work.put(jid)
    return job


def public(job: dict) -> dict:
    out = {"id": job["id"], "status": job["status"], "filename": job.get("filename", "")}
    if job["status"] == "queued":
        out["position"] = sum(1 for j in jobs.values() if j["status"] == "queued" and j["created"] <= job["created"])
    if job["status"] == "done":
        out.update(job["result"])
    if job["status"] == "error":
        out["error"] = job.get("error", "")
    return out


def wait_for(job: dict, seconds: int) -> dict:
    ev = events.get(job["id"])
    if ev:
        ev.wait(seconds)
    return job


@app.get("/health")
def health():
    ready = _model_ready.is_set() and (_model is not None or FAKE_MODEL)
    return {"ok": ready, "status": "готов" if ready else ("ошибка модели" if _model_error else "загружаю модель"),
            "model": MODEL, "queue": sum(1 for j in jobs.values() if j["status"] in ("queued", "running")),
            "version": VERSION, **({"error": _model_error} if _model_error else {})}


@app.post("/v1/transcribe")
def transcribe(request: Request, file: UploadFile = File(...), language: str = Form(""), speakers: str = Form("auto"),
               names: str = Form(""), prompt: str = Form(""), wait: str = Form("1")):
    who = auth(request)
    if not who or who == "__blocked__":
        return deny(who)
    if speakers not in ("auto", "mono", "stereo"):
        speakers = "auto"
    try:
        job = new_job(file, who, {"language": language[:5], "speakers": speakers, "names": names[:100], "prompt": prompt[:500]})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=413)
    if wait in ("0", "false", "no"):
        return JSONResponse(public(job), status_code=202)
    wait_for(job, WAIT_SECONDS)
    data = public(job)
    return JSONResponse(data, status_code=200 if job["status"] == "done" else (500 if job["status"] == "error" else 202))


@app.get("/v1/jobs/{jid}")
def get_job(jid: str, request: Request):
    who = auth(request)
    if not who or who == "__blocked__":
        return deny(who)
    job = jobs.get(jid)
    if not job or job["client"] != who:
        return JSONResponse({"error": "Задание не найдено"}, status_code=404)
    return public(job)


def fmt_ts(t: float, sep: str) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s - int(s)) * 1000):03d}"


@app.post("/v1/audio/transcriptions")
def openai_compatible(request: Request, file: UploadFile = File(...), model: str = Form("whisper-1"),
                      language: str = Form(""), prompt: str = Form(""), response_format: str = Form("json")):
    who = auth(request)
    if not who or who == "__blocked__":
        return deny(who)
    try:
        job = new_job(file, who, {"language": language[:5], "speakers": "mono", "prompt": prompt[:500]})
    except ValueError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=413)
    wait_for(job, WAIT_SECONDS)
    if job["status"] != "done":
        return JSONResponse({"error": {"message": job.get("error") or "Не успели расшифровать, файл слишком длинный"}}, status_code=500)
    r = job["result"]
    if response_format == "text":
        return PlainTextResponse(r["text"])
    if response_format in ("srt", "vtt"):
        sep = "," if response_format == "srt" else "."
        blocks = []
        for i, s in enumerate(r["segments"], 1):
            blocks.append((f"{i}\n" if response_format == "srt" else "") + f"{fmt_ts(s['start'], sep)} --> {fmt_ts(s['end'], sep)}\n{s['text']}\n")
        return PlainTextResponse(("WEBVTT\n\n" if response_format == "vtt" else "") + "\n".join(blocks))
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": r["language"], "duration": r["duration"], "text": r["text"],
                "segments": [dict(id=i, **s) for i, s in enumerate(r["segments"])]}
    return {"text": r["text"]}


@app.get("/v1/stats")
def stats(request: Request):
    who = auth(request)
    if not who or who == "__blocked__":
        return deny(who)
    try:
        return json.loads((DATA_DIR / "stats.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


@app.get("/", response_class=HTMLResponse)
def page():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})
