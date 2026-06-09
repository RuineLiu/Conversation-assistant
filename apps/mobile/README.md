# Proactive Assistant Mobile

Flutter v0 client for the product backend.

## Current Scope

This first scaffold supports the manual transcript product flow:

- configure backend base URL
- create a session
- append transcript text
- display transcript and meeting state counts
- display proactive prompts
- record accept / reject / open-detail feedback
- generate and view session summary
- end a session

Audio recording is intentionally not wired yet. The backend already exposes:

```text
POST /asr/transcribe
POST /sessions/{session_id}/audio-transcript
```

Those endpoints can be connected after the real Azure Speech smoke test and Flutter recording package decision.

## Setup

Native iOS and Android platform folders have been generated with Flutter. Install dependencies and run checks with:

```bash
cd /Users/jorahmormont/Desktop/proactive_work/apps/mobile
flutter pub get
flutter analyze
flutter test
```

On macOS, Flutter may require accepting the Xcode license first:

```bash
sudo xcodebuild -license
```

Run the backend:

```bash
cd /Users/jorahmormont/Desktop/proactive_work
uv run uvicorn proactive_assistant.product.api:app --reload --host 0.0.0.0 --port 8001
```

Run the app:

```bash
cd /Users/jorahmormont/Desktop/proactive_work/apps/mobile
flutter run
```

For iOS Simulator, `http://127.0.0.1:8001` should work. For Android Emulator, use:

```text
http://10.0.2.2:8001
```
