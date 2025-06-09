# planner_agent.py using LLaMA 2 13B Chat with correct JSON extraction, matching, and debug prints

import json
import re
import torch
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM

class PlannerAgent:
    def __init__(self, class_map_path):
        print("[INIT] Loading class map and model...")
        with open(class_map_path) as f:
            self.class_map = json.load(f)

        model_name = "meta-llama/Llama-2-13b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=True
        )
        print("[INIT] Model and tokenizer loaded successfully.")

    def match_object_to_class(self, obj_str):
        obj_str = obj_str.strip().lower()
        class_keys = [k.lower() for k in self.class_map.keys()]
        matches = difflib.get_close_matches(obj_str, class_keys, n=1, cutoff=0.5)
        if matches:
            matched_key = matches[0]
            for original_key in self.class_map.keys():
                if original_key.lower() == matched_key:
                    print(f"[MATCH] Matched '{obj_str}' to class '{original_key}'")
                    return original_key
        print(f"[MATCH] No match found for object: {obj_str}")
        return None

    def run(self, user_prompt):
        options = ', '.join([f'"{k}"' for k in self.class_map.keys()])

        prompt = f"""
<s>[INST] You are an assistant that extracts an object and a spatial zone from the user query.

Choose the object from this list: [{options}]
Return only valid JSON like: {{ "object": "bus", "zone": "center" }}

User query: "{user_prompt}" [/INST]
"""

        print("[PROMPT] Sending to model:")
        print(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("[MODEL OUTPUT]")
        print(result)

        try:
            matches = re.findall(r"\{\s*\"object\".*?\}", result)
            if not matches:
                raise ValueError("No JSON object found in output")
            parsed = json.loads(matches[-1])
            print("[PARSED JSON]")
            print(parsed)
        except Exception:
            raise ValueError("Failed to parse LLaMA response:\n" + result)

        matched_obj = self.match_object_to_class(parsed["object"])
        if not matched_obj:
            raise ValueError(f"No match found for object: {parsed['object']}")

        parsed["object"] = matched_obj
        parsed["class_id"] = self.class_map[matched_obj]
        print("[FINAL OUTPUT]")
        print(parsed)
        return parsed
