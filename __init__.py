import os
import json
from zai import ZhipuAiClient, core as zai_core

# Function to load the API key from a config file
def get_api_key():
    # Construct the path to the config file, assuming it's in the same directory as this script
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    # Check if config file exists and load the key
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            api_key = config.get("api_key")
            if api_key and api_key != "YOUR_ZHIPU_API_KEY_HERE":
                return api_key
        except json.JSONDecodeError:
            # Log an error if JSON is malformed
            print("[AI Prompt Cleaner] Error: config.json is malformed.")
            return None
    
    # Return None if file doesn't exist or key is not found/set
    return None

class SafePrompt:
    def __init__(self):
        # Load the API key during node initialization
        self.api_key = get_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input parameters for the node.
        """
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "instruction": ("STRING", {"multiline": True, "default": "Remove all NSFW content."}),
                "model_name": (["glm-4.5-flash", "glm-z1-flash", "glm-4.5","glm-4.5-x","glm-4.5-airx"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_prompt",)
    FUNCTION = "execute"
    CATEGORY = "SafePrompt"

    def execute(self, prompt, instruction, model_name, seed):
        if not self.api_key:
            raise ValueError("Zhipu AI API Key is not configured. Please create a 'config.json' file in the node's directory and add your key.")

        system_prompt = "你是一个专业的提示词工程师。请根据用户要求，修改、清洗或重写给定的文本，需要遵循以下原则：1. 严格按照用户的具体要求进行修改; 2. 保持修改的合理性和逻辑性; 3. 你必须只返回修改后的文本结果绝不添加任何额外的解释、说明、前言、致谢或任何与结果无关的内容; 4. 保持原文的核心信息和语言风格（除非用户明确要求改变);"

        user_prompt = f"原始文本如下：\n---\n{prompt}\n---\n\n我的要求是：\n---\n{instruction}\n---\n\n请返回修改后的文本："

        try:
            client = ZhipuAiClient(api_key=self.api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                seed=seed,
                temperature=0.1
            )
            
            cleaned_text = response.choices[0].message.content
            return (cleaned_text.strip(),)
        except zai_core.APIStatusError as err:
            error_message = f"Zhipu AI API Status Error: {err}"
            print(f"[AI Prompt Cleaner] {error_message}")
            raise ValueError(error_message)
        except zai_core.APITimeoutError as err:
            error_message = f"Zhipu AI API Timeout Error: {err}"
            print(f"[AI Prompt Cleaner] {error_message}")
            raise TimeoutError(error_message)
        except Exception as err:
            error_message = f"An unexpected error occurred: {err}"
            print(f"[AI Prompt Cleaner] {error_message}")
            raise Exception(error_message)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SafePrompt": SafePrompt
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SafePrompt": "SafePrompt"
}