# data/

Git ignores all files in this folder except this README. Do not commit videos, audio, frames or transcripts.

## 1. Synthetic sample folders (default, no download)

`vid2thumb make-sample --out data/sample_bread --topic bread` writes one sample folder.
The tests and `vid2thumb demo` make the same folders in a temporary directory.

| File | Contents |
|---|---|
| `audio.wav` | 16 kHz mono tone bursts (a stand-in for speech) with silences of 0.4 s and 1.5 s |
| `frames/frame_00001.png` … | 1 fps frames, 320 × 180, three scenes, one blurred frame |
| `transcript.srt` | Synthetic auto-captions with seeded word errors (default 8 %) |
| `reference.json` | Exact transcript, reference summary, scene starts, index of the blurred frame |

Topics: `bread` (a sourdough recipe) and `bicycle` (a flat-tyre repair). All text was written for this project.

## 2. Your own videos

| Item | Rule |
|---|---|
| Rights | Use only videos that you own or that have a licence that allows download and reuse (for example Creative Commons) |
| URL input | `vid2thumb run --url ... ` downloads only if the licence is in the allow list, or if you give `--i-own-this` |
| Length | At most `VID2THUMB_MAX_MINUTES` minutes (default 30) |
| Transcript | Optional sidecar file with the same name as the video: `talk.mp4` + `talk.srt` (or `.vtt`, `.json`, `.txt`) |
| Reference | Optional `talk.reference.json` with `transcript` and `summary` to get WER and ROUGE |
| Program | `ffmpeg` on PATH, to get audio and frames from a video file |

## 3. Evaluation set (planned)

Collect 10 to 20 short licensed videos. For each video, write a checked transcript and a
reference summary in a `.reference.json` file. List the videos in a JSONL manifest:

```json
{"input": "data/videos/talk01.mp4"}
```

Then run `vid2thumb evaluate --manifest data/manifest.jsonl`.
