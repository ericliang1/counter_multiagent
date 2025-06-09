import cv2
import json
from ultralytics import YOLO

class LineDrawerAgent:
    def __init__(self, model_path, midpoint, class_id=None):
        self.model = YOLO(model_path)
        self.midpoint = midpoint  # (x, y)
        self.class_id = class_id  # Optional filter by object class

    def _average_motion_vector(self, vectors):
        if not vectors:
            return (1, 0)
        avg_dx = sum(v[0] for v in vectors.values()) / len(vectors)
        avg_dy = sum(v[1] for v in vectors.values()) / len(vectors)
        return (avg_dx, avg_dy)

    def _get_perpendicular_vector(self, dx, dy):
        return -dy, dx

    def _draw_perpendicular_line(self, frame, direction, length=250, color=(0, 255, 0), thickness=3):
        dx, dy = self._get_perpendicular_vector(*direction)
        x0, y0 = self.midpoint
        x1 = int(x0 + dx * length)
        y1 = int(y0 + dy * length)
        x2 = int(x0 - dx * length)
        y2 = int(y0 - dy * length)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        return (x1, y1), (x2, y2)

    def run(self, video_path, save_path="output/gate_line.json", display=False):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(fps * 10)
        frame_count = 0

        prev_centers = {}
        motion_vectors = {}
        line_start, line_end = None, None

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = self.model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes

            for box in detections:
                if box.id is None:
                    continue
                if self.class_id is not None and int(box.cls.item()) != self.class_id:
                    continue

                id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                if id in prev_centers:
                    prev = prev_centers[id]
                    dx = center[0] - prev[0]
                    dy = center[1] - prev[1]
                    motion_vectors[id] = (dx, dy)
                prev_centers[id] = center

                if display:
                    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
                    if id in motion_vectors:
                        dx, dy = motion_vectors[id]
                        arrow_end = (int(center[0] + dx * 5), int(center[1] + dy * 5))
                        cv2.arrowedLine(frame, (int(center[0]), int(center[1])), arrow_end, (0, 0, 255), 2)
                    cv2.putText(frame, f"ID {id}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw midpoint marker
            if display:
                cv2.circle(frame, self.midpoint, 8, (0, 255, 0), -1)
                cv2.putText(frame, "Midpoint", (self.midpoint[0] + 10, self.midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw line
            avg_vector = self._average_motion_vector(motion_vectors)
            line_start, line_end = self._draw_perpendicular_line(frame, avg_vector)

            if display:
                cv2.imshow("LineDrawerAgent Output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        # Save line to JSON
        if line_start and line_end:
            with open(save_path, "w") as f:
                json.dump({"line_start": line_start, "line_end": line_end}, f)
            print(f"✅ Gate line saved to {save_path}")
        else:
            print("❌ No line was drawn due to insufficient motion data.")
