# Voicebot

Browser mic -> **Whisper** (STT) -> **gpt-4o-mini** -> **Deepgram Aura** (TTS) -> browser speaker.

```
voicebot/
├── api/
│   └── index.py          # FastAPI app (Vercel serverless function)
├── public/
│   └── index.html        # UI (served statically)
├── requirements.txt
├── requirements-dev.txt  # adds uvicorn for local runs
├── vercel.json
└── .env.example
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env        # then fill in your keys
export $(grep -v '^#' .env | xargs)   # Windows PowerShell: set $env:OPENAI_API_KEY="..." etc.

uvicorn api.index:app --reload --port 8000
```

Open http://localhost:8000 (mic access works on `localhost`).

## Deploy to Vercel

1. Push this folder to GitHub (the `.env` file is git-ignored).
2. In Vercel: **Add New -> Project**, import the repo. No build settings needed.
3. Under **Settings -> Environment Variables** add `OPENAI_API_KEY` and `DEEPGRAM_API_KEY`.
4. Deploy. (Or use the CLI: `npm i -g vercel && vercel --prod`.)

Mic access requires HTTPS, which Vercel provides by default.

## Notes

- **Limits:** Vercel request bodies are capped at ~4.5 MB; the UI stops recording at 60 s, which stays well under that. `maxDuration` is set to 60 s in `vercel.json`; lower it if your plan caps functions at less.
- **Voice:** change `DEEPGRAM_VOICE` (e.g. `aura-2-andromeda-en`, `aura-2-apollo-en`).
- **Language:** set `WHISPER_LANGUAGE` (e.g. `hi`) to skip auto-detection. Note that the Deepgram Aura voices are English-first, so match the voice to your language.
- **Health check:** `GET /api/health` reports whether both keys are configured.

## Logging

The API logs to stdout with the `voicebot` logger (Vercel captures this automatically):

- Every request: `REQUEST <METHOD> <path> -> <status> (<ms>)`
- Voice pipeline stages: `STT starting/done/failed`, `Chat starting/done/failed`, `TTS starting/done/failed`
- Failures use `logger.exception`, so tracebacks land in the logs
- Service startup line with the configured models

- **Level:** defaults to `INFO`. Set `LOG_LEVEL=DEBUG` (env var on Vercel or in `.env`) for full detail.
- **View in Vercel:** `vercel logs <deployment-url>` or the **Logs** tab of the deployment in the Vercel dashboard. Function logs appear per-invocation under the serverless function.
- **What to look for when debugging:** the `REQUEST` line tells you the status/latency; the pipeline-stage lines tell you which step failed (STT / Chat / TTS) and how long it took. Logs never include raw audio or API keys.

