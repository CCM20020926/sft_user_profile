from config import get_llm_client
from langchain.schema import SystemMessage, HumanMessage
from prompt import UserProfileAgentPrompt

class UserProfileAgent:
    def __init__(self, model, provider):
        self.llm = get_llm_client(model, provider)
        self.prompt_template = UserProfileAgentPrompt.load_prompt_template()
    
    def generate_user_profile(self, data: str) -> str:
        prompt = self.prompt_template.format(data=data)
        response = self.llm.invoke([
                SystemMessage(UserProfileAgentPrompt.system_prompt),
                HumanMessage(prompt)
            ]
        )

        return response


