import google.generativeai as genai

class ResponseChecker():
    def __init__(self):
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def is_response_dangerous(self, response):
        to_check = f"Is this safe for children: \"\"\"\n{response}\"\"\". Say yes or no"
        check = self.model.generate_content(to_check)
        print("Check text =", check.text)

        if "No" in check.text or "no" in check.text:
            return True
        
        return False
        
