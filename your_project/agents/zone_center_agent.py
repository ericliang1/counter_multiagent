import cv2
import re
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
 
class ZoneCenterAgent:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

    def run(self, image_path, prompt="Where is the area of interest in this image?"):
        image = Image.open(image_path)
        width, height = image.size

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                        f"{prompt} Return it as a bounding box with four normalized values (x1, y1, x2, y2)."
                    )},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=100)
        response = self.processor.decode(output[0], skip_special_tokens=True)

        match = re.search(r"\[\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\]", response)
        if match:
            x1, y1, x2, y2 = map(float, match.groups())
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return int(center_x * width), int(center_y * height)

        raise ValueError(f"❌ Could not extract box from response: {response}")
