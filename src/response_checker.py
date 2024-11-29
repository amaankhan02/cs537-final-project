import google.generativeai as genai
from deprecated import deprecated

@deprecated(reason="Deprecated because we are no longer using a response checker")
class ResponseChecker():
    def __init__(self):
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def is_response_safe(self, response):
        to_check = f"Is this safe for children: \"\"\"\n{response}\"\"\". Say yes or no. Do not add any extra explanation"
        check = self.model.generate_content(to_check)

        check_text = check.text.lower()

        if ("no" in check_text and check_text.index("no") == 0):
            return False
        
        return True
        
