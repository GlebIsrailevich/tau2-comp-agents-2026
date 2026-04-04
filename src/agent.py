import os

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from messenger import Messenger

load_dotenv()

# Yandex GPT configuration (default)
YA_GPT_FOLDER_ID = os.getenv("YA_GPT_FOLDER_ID", "")
YA_GPT_AUTH = os.getenv("YA_GPT_AUTH", "")

# Generic OpenAI-compatible configuration (fallback)
AGENT_LLM = os.getenv("AGENT_LLM", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


class Agent:
    def __init__(self):
        self.messenger = Messenger()

        # Configure LLM based on available environment variables
        if YA_GPT_FOLDER_ID and YA_GPT_AUTH:
            # Yandex GPT configuration
            self.llm = ChatOpenAI(
                model=f"gpt://{YA_GPT_FOLDER_ID}/gpt-oss-20b/latest",
                openai_api_key=YA_GPT_AUTH,
                openai_api_base="https://llm.api.cloud.yandex.net/v1",
                temperature=0.1,
                timeout=60,
            )
        elif AGENT_LLM:
            # Generic OpenAI-compatible configuration
            model_name = AGENT_LLM.split("/")[-1] if "/" in AGENT_LLM else AGENT_LLM
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                temperature=0.1,
                timeout=60,
            )
        else:
            raise ValueError(
                "No LLM configuration provided. Set either YA_GPT_FOLDER_ID/YA_GPT_AUTH "
                "or AGENT_LLM/OPENAI_API_KEY environment variables."
            )

        self.history: list = [
            SystemMessage(
                content=(
                    "You are a customer service agent. "
                    "You will receive a system prompt describing available tools and policy. "
                    "Always respond with a single valid JSON object with keys 'name' and 'arguments'. "
                    "Never include any text outside the JSON."
                )
            )
        ]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        self.history.append(HumanMessage(content=input_text))
        response = self.llm.invoke(self.history)
        reply = response.content
        self.history.append(AIMessage(content=reply))

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )
