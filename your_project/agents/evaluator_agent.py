import cv2
import json
from ultralytics import YOLO

class EvaluatorAgent:
    def __init__(self, model_path, line_path, class_id):
        self.model = YOLO(model_path)
        self.class_id = class_id
        with open(line_path, "r") as f:
            data = json.load(f)
        self.line_start = tuple(data["line_start"])
        self.line_end = tuple(data["line_end"])

    def _did_cross_line(self, prev, curr):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        A, B = self.line_start, self.line_end
        return (
            ccw(prev, A, B) != ccw(curr, A, B) and
            ccw(prev, curr, A) != ccw(prev, curr, B)
        )

    def run(self, video_path, output_path="output/output_video.mp4"):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        prev_centers = {}
        counted_ids = set()
        object_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                iou=0.3,
                conf=0.4
            )
            detections = results[0].boxes

            for box in detections:
                if box.id is None or int(box.cls.item()) != self.class_id:
                    continue

                id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                if id in prev_centers:
                    prev = prev_centers[id]
                    if id not in counted_ids and self._did_cross_line(prev, center):
                        object_count += 1
                        counted_ids.add(id)
                        print(f"Object {id} crossed the line. Total count: {object_count}")

                prev_centers[id] = center

                # Draw object center and ID
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"ID {id}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw gate line and count
            cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 3)
            cv2.putText(frame, f"Count: {object_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            cv2.imshow("EvaluatorAgent Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"✅ Output video saved to: {output_path}")
        return object_count
