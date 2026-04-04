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
        self._llm = None
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

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """Create LLM instance based on available configuration."""
        if YA_GPT_FOLDER_ID and YA_GPT_AUTH:
            return ChatOpenAI(
                model=f"gpt://{YA_GPT_FOLDER_ID}/gpt-oss-20b/latest",
                openai_api_key=YA_GPT_AUTH,
                openai_api_base="https://llm.api.cloud.yandex.net/v1",
                temperature=0.1,
                timeout=60,
            )
        elif AGENT_LLM:
            model_name = AGENT_LLM.split("/")[-1] if "/" in AGENT_LLM else AGENT_LLM
            return ChatOpenAI(
                model=model_name,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                temperature=0.1,
                timeout=60,
            )
        return None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        self.history.append(HumanMessage(content=input_text))

        # Handle case when LLM is not configured
        if self.llm is None:
            print("Warning: No LLM configured, returning fallback response")
            reply = '{"name": "greeting", "arguments": {"message": "Hello! How can I help you?"}}'
            self.history.append(AIMessage(content=reply))
        else:
            try:
                response = await self.llm.ainvoke(self.history)
                reply = response.content
                self.history.append(AIMessage(content=reply))
            except Exception as e:
                # Fallback response for API errors
                print(f"LLM API error: {e}")
                reply = '{"name": "error", "arguments": {"message": "Service temporarily unavailable"}}'
                self.history.append(AIMessage(content=reply))

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )
