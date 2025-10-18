
class UserProfileAgentPrompt:

    system_prompt = """You are a helpful assistant for user profiling. You should generate a user profile according to given profile information.
You should generate a structured output which strictly follows required format, and do not return any addition texts.
"""

    @staticmethod
    def load_prompt_template():
        with open('./prompt_template.md', 'r') as f:
            return f.read()
