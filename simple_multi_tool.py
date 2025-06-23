from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager


class DebugCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n🔍 LLM START")
        for prompt in prompts:
            print("📥 Prompt:\n", prompt)

    def on_llm_end(self, response, **kwargs):
        print("✅ LLM END")
        print("📤 Output:\n", response.generations[0][0].text)

    def on_tool_start(self, tool, input_str, **kwargs):
        print(f"🛠️ Tool `{tool}` aufgerufen mit:\n{input_str}")

    def on_tool_end(self, output, **kwargs):
        print("🔚 Tool-Antwort:\n", output)

    def on_chain_end(self, outputs, **kwargs):
        print("🧾 Finaler Output:", outputs)


# 🔧 LLM von Ollama
llm = ChatOllama(
    model="llama3.2:3b",  # llama3:instruct llama3.2:3b, llama3.2:1b
    base_url="http://localhost:11434",  # falls Ollama in Docker läuft
    temperature=0,
    verbose=True,
    callback_manager=CallbackManager([DebugCallbackHandler()])
)

# 🧰 Tools definieren
def get_weather(location: str) -> str:
    print(f"🔍 Wetterabfrage für: {location}")
    return f"In {location} ist es sonnig und warm. Du kannst den Spaziergang planen."

def check_calendar(date: str) -> str:
    print(f"🔍 Kalenderabfrage für: {date}")
    return f"Dein Kalender ist am {date} frei. Du kannst den Termin setzen."


tools = [
    Tool(name="GetWeather", func=get_weather, description="Gibt das Wetter für einen Ort zurück. Eingabe: Ort"),
    Tool(name="CheckCalendar", func=check_calendar, description="Überprüft, ob ein Termin an einem Tag frei ist. Eingabe: Datum")
]
# 🤖 Agent mit Ollama & strukturierter Tool-Steuerung
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=5,  # Limitiert Wiederholungen
    callback_manager=CallbackManager([DebugCallbackHandler()])  # Optional für Debugging
)

# Prompt
prompt = (
    "Wie ist das Wetter in München? "
    "Wenn es schön ist, prüfe ob ich morgen Zeit habe. "
)

# invoke statt run
result = agent.invoke({"input": prompt})
print("\n🔚 Antwort:\n", result["output"] if isinstance(result, dict) else result)