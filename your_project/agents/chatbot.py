from planner_agent import PlannerAgent
from zone_center_agent import ZoneCenterAgent
from line_drawer_agent import LineDrawerAgent
from evaluator_agent import EvaluatorAgent

import cv2

def simulate_agent_response(role, message):
    print(f"\n[{role.upper()}] {message}")

def main_chat_loop():
    model_path = "models/yolov8m.pt"
    class_map_path = "data/class_map.json"
    video_path = "data/videos/input.mp4"
    frame_path = "data/frame.jpg"
    gate_line_path = "output/gate_line.json"

    while True:
        user_input = input("\n[USER] What do you want to count? (or type 'exit')\n> ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Step 1: Interpret user prompt
        planner = PlannerAgent(class_map_path)
        task = planner.run(user_input)
        simulate_agent_response("PlannerAgent", f"Detected object: {task['object']}, class ID: {task['class_id']}, zone: {task['zone']}")

        # Step 2: Capture frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frame_path, frame)
        cap.release()

        # Step 3: Find center of zone
        zone_center = ZoneCenterAgent()
        center = zone_center.run(frame_path, prompt=f"Where is the {task['zone']}?")
        simulate_agent_response("ZoneCenterAgent", f"Estimated center of {task['zone']} at {center}")

        # Step 4: Draw motion-based gate line
        line_drawer = LineDrawerAgent(model_path, center, task["class_id"])
        line_drawer.run(video_path)
        simulate_agent_response("LineDrawerAgent", f"Gate line drawn at center {center} and saved to {gate_line_path}")

        # Step 5: Count objects
        evaluator = EvaluatorAgent(model_path, gate_line_path, task["class_id"])
        count = evaluator.run(video_path)
        simulate_agent_response("EvaluatorAgent", f"{count} {task['object']}(s) crossed the zone.")

if __name__ == "__main__":
    main_chat_loop()
