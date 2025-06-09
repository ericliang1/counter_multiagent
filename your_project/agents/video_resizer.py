import cv2
import sys

def resize_video(input_path, output_path, width, height):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        out.write(resized)

    cap.release()
    out.release()
    print(f"✅ Resized video saved to: {output_path}")

if __name__ == "__main__":
    input_path = "data/videos/original.mp4"
    output_path = "data/videos/input.mp4"
    width = 1280
    height = 720

    resize_video(input_path, output_path, width, height)