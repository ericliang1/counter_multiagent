A natural-language-driven object counting pipeline for video. Tell it what to count and roughly where — e.g. *"How many pass through the center of the road?"* — and it locates the region, drops a virtual counting line, tracks objects with YOLOv8, and tallies how many cross.

> Developed completely by Eric Liang under the guidance of Kaiwen Yuan. Presented to a Safari AI all-hands meeting and concluded in May 2025.

## What it's for

The goal is a counting system that **adapts to any camera angle automatically**. Traditional video counters need a human to manually draw a counting line or polygon for each new camera, and to re-do it whenever the view changes. That doesn't scale across many cameras or shifting deployments.

Here, the placement is inferred instead of hand-drawn: a vision model finds the region from a plain-English description, and the gate line is oriented from the actual motion seen in that scene. So the same prompt — *"how many pass through the center of the road?"* — works whether the camera is mounted overhead, at a steep oblique angle, or off to the side. The system effectively **self-calibrates its counting geometry per scene**, which is what makes it deployable across arbitrary angles without manual setup.

## How it works

```
User prompt
   │  "How many pass through the center of the road?"
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

The **Planner** reads your prompt and turns it into a structured task: an object class drawn from the 80 COCO categories (with fuzzy-matching, so `"buss"` becomes `"bus"`) plus a free-text zone like `"center of the road"`. The first frame of the video is then saved as an image for grounding.

The **ZoneCenter** agent looks at that frame and translates the vague zone phrase into a single pixel coordinate — the anchor for where the counting line will sit.

The **LineDrawer** tracks objects of the target class for about ten seconds, estimates their average direction of motion, and draws a "gate" line through the anchor point oriented perpendicular to that motion — so objects travelling through the zone cross it head-on. This is the step that makes the system angle-agnostic: the line follows the scene's own flow rather than a fixed orientation.

The **Evaluator** then tracks objects across the full video and counts each unique track exactly once as its center crosses the gate, using a counterclockwise segment-intersection test. It writes an annotated video with the gate, track IDs, and a running count.
