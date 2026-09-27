# BanjirKita AI — 120-Second Video Script

**Target runtime: 115 seconds** (5s buffer under the 120s limit — better to come in slightly under than risk a penalty)
**Language:** English (no subtitles needed)
**Pace assumption:** ~130 words/minute, clear deliberate speech

---

## Filming Checklist (before you hit record)

- [ ] Real voice — no AI-generated voiceover (explicitly banned)
- [ ] You speaking directly to camera for at least part of the video (required)
- [ ] File format: MP4, WMV, FLV, or AVI
- [ ] File size ≤ 60 MB (competition limit) — compress if needed
- [ ] Stay under 2:00 total — 115s target leaves you a safety margin
- [ ] If you switch to Bahasa Malaysia/Mandarin at any point, add English subtitles for that portion

---

## 1. Self Introduction — `0:00–0:10` (10s)

**[Camera: you, speaking to camera]**

> "Hi, I'm [Name], a computing student at Universiti Malaysia Sabah. This is BanjirKita AI — a community flood intelligence system for Malaysia's monsoon season."

*(~26 words)*

---

## 2. Problem Statement — `0:10–0:28` (18s)

**[Camera: you, speaking to camera]**

> "Every year, monsoon floods displace over a hundred thousand Malaysians. In flood-prone states like Kelantan and Terengganu, rural communities often get under an hour of warning — because sensors are sparse, and networks fail right when they're needed most."

*(~42 words)*

---

## 3. Project/Solution Description — `0:28–0:50` (22s)

**[Camera: you, speaking to camera]**

> "BanjirKita AI turns residents' own smartphones into privacy-preserving flood sensors. An on-device AI model reads local photos, text reports, and rainfall data to estimate flood risk — completely offline. Raw images and locations never leave the phone. Only anonymized model updates are shared, so the system gets smarter as more households join."

*(~55 words)*

---

## 4. Demonstration — `0:50–1:20` (30s)

**[Camera cuts to: screen recording of laptop]**

> "Here's a live inference run. I feed in a sample flood photo and rainfall reading —"

**[On-screen: result/risk score appears on screen]**

> "— and in under a second, the on-device model returns a risk level and confidence score."

**[On-screen: simple two-phone diagram, arrow between them labeled "model weights only — no photos, no location"]**

> "This shows how devices improve the shared model together — without ever exchanging personal photos or locations."

*(~60 words)*

**Note:** You don't need to actually run federated learning across two devices for this — a single-device inference demo (real, on-screen, running) plus this static/simple diagram is enough to prove it's not just a mockup. Don't over-engineer the demo for the time you have.

---

## 5. Technology Used — `1:20–1:40` (20s)

**[Camera: you, speaking to camera — or voiceover-your-own-voice over a slide with Intel logos/architecture]**

> "The inference pipeline runs on an Intel Arc A770 GPU, optimized with Intel OpenVINO — built to run on power-efficient Intel NPU hardware in the field, so it keeps working even when the internet goes down."

*(~36 words)*

**Note:** Say this honestly — Arc A770 is what you actually used to build and optimize the pipeline; NPU is the deployment target, not something you've tested yet. Don't imply it's already running on NPU hardware.

---

## 6. Future Scope — `1:40–1:55` (15s)

**[Camera: you, speaking to camera]**

> "Our roadmap starts with a pilot in one Kelantan village through the local disaster committee, then expands statewide — with the same architecture adaptable to any flood-prone region in Southeast Asia."

*(~32 words)*

---

## Total

- **Spoken words:** ~249
- **Estimated runtime:** ~115 seconds
- **Sections covered:** all 6 required points ✓

## B-roll / Screen Assets to Prepare Beforehand

1. Laptop screen recording: sample photo + rainfall number → model output (risk level + confidence score)
2. Simple diagram (can be a slide, doesn't need animation): two phone icons + arrow labeled "model weights only"
3. Optional: one slide with Intel Arc / OpenVINO logos for the tech section

## Delivery Tips

- Practice the problem + solution sections out loud once or twice — they're the densest (42 and 55 words in ~18–22s), easy to rush.
- Record the demo screen capture *separately* from your talking-head footage, then edit them together — much easier than trying to talk and operate the laptop at the same time.
- If you go slightly over on one section, you have ~5s of buffer built in (target is 115s, limit is 120s) — don't panic-cut mid-sentence.
