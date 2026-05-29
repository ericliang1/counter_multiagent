A natural-language-driven object counting pipeline for video. Tell it what to count and roughly where — e.g. *"How many buses pass through the center?"* — and it locates the region, drops a virtual counting line, tracks objects with YOLOv8, and tallies how many cross.

> Developed under the guidance of Kaiwen Yuan. Presented to a Safari AI all-hands meeting and concluded in May 2025.

## How it works

```
User prompt
   │  "How many buses pass through the center?"
   ▼
Planner (Llama 2 13B)         →  { object, zone, class_id }
   │  first frame saved to data/frame.jpg
   ▼
ZoneCenter (LLaVA 1.5 7B)     →  pixel point (x, y)
   │
   ▼
LineDrawer (YOLOv8m)          →  gate line perpendicular to motion
   │
   ▼
Evaluator (YOLOv8m)           →  count + annotated output video
```

Four roles run in sequence as plain Python classes in `chatbot.py` — the "multi-agent" label means role separation, not an agent framework.

The **Planner** reads your prompt and turns it into a structured task: an object class drawn from the 80 COCO categories (with fuzzy-matching, so `"buss"` becomes `"bus"`) plus a free-text zone like `"center"` or `"left side"`. The first frame of the video is then saved as an image for grounding.

The **ZoneCenter** agent looks at that frame and translates the vague zone phrase into a single pixel coordinate — the anchor for where the counting line will sit.

The **LineDrawer** tracks objects of the target class for about ten seconds, estimates their average direction of motion, and draws a "gate" line through the anchor point oriented perpendicular to that motion — so objects travelling through the zone cross it head-on.

The **Evaluator** then tracks objects across the full video and counts each unique track exactly once as its center crosses the gate, using a counterclockwise segment-intersection test. It writes an annotated video with the gate, track IDs, and a running count.

The core idea: language supplies the *what* and a rough *where*, while motion and detection supply the line geometry and the counting. Conceptually it's a virtual tripwire — no manual class picking or line drawing required.
