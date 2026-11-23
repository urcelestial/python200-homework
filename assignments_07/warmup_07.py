"""Week 7 warmups: tool-based agents, ReAct, and smolagents."""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {"celsius": {"type": "number", "description": "Temperature in degrees Celsius."}},
            "required": ["celsius"],
            "additionalProperties": False,
        },
    },
}

for temperature in (0, 100, -40):
    print(celsius_to_fahrenheit(temperature))


def get_current_time() -> str:
    """Return the current local time as a string."""
    from datetime import datetime

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def run_agent(user_query: str, tools=None, dispatch=None, max_rounds: int = 5) -> str:
    """Run a small tool-calling ReAct loop and return the assistant's final text."""
    tools = tools or [get_current_time_schema]
    dispatch = dispatch or {"get_current_time": lambda _: get_current_time()}
    messages = [{"role": "user", "content": user_query}]
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or ""
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = dispatch[tool_call.function.name](arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
    return "Stopped after reaching the tool-round limit."


# Q2 prediction: No. The only available tool returns the current time, so it
# cannot help with the conversion. I expect one API call, with the model answering
# directly from its own knowledge and arithmetic.
response = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Q2 response:", response)
# The returned text lets me compare the actual run with that prediction.


# Q3
TOOLS = [get_current_time_schema, celsius_to_fahrenheit_schema]
NODE_TOOLS = {
    "get_current_time": lambda _: get_current_time(),
    "celsius_to_fahrenheit": lambda args: celsius_to_fahrenheit(args["celsius"]),
}
response_a = run_agent(
    "What is 37 degrees Celsius in Fahrenheit?", tools=TOOLS, dispatch=NODE_TOOLS
)
print("Response A:", response_a)
# The conversion tool should be called because it is an explicit temperature conversion.
response_b = run_agent(
    "What is the boiling point of water in plain English?", tools=TOOLS, dispatch=NODE_TOOLS
)
print("Response B:", response_b)
# No tool is needed: the model can answer this general knowledge question directly.


# --- Lesson 03: Multi-Tool Agent ---

class CsvManager:
    """Load and inspect a CSV file for an agent."""

    def __init__(self):
        self.df = None

    def load_csv(self, filename: str) -> dict:
        """Load a CSV from resources and return its shape and column names."""
        path = Path("resources") / filename
        if not path.is_file():
            return {"error": f"CSV file not found: {path}"}
        self.df = pd.read_csv(path)
        return {"shape": self.df.shape, "columns": list(self.df.columns)}

    def get_column_stats(self, column: str) -> dict:
        """Return descriptive statistics for a loaded column."""
        if self.df is None:
            return {"error": "No CSV is loaded."}
        if column not in self.df.columns:
            return {"error": f"Column not found: {column}"}
        return self.df[column].describe().to_dict()

    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.

        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV is loaded."}
        missing = [column for column in (col1, col2) if column not in self.df.columns]
        if missing:
            return {"error": f"Column not found: {', '.join(missing)}"}
        try:
            r_value, p_value = pearsonr(self.df[col1].dropna(), self.df[col2].dropna())
        except (TypeError, ValueError) as error:
            return {"error": str(error)}
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r_value), 4),
            "p_value": round(float(p_value), 4),
        }


csv_manager = CsvManager()
SYSTEM_PROMPT = """You are a data assistant. Use tools to load and inspect CSV data.
When a task requires a calculation, use the relevant tool. Explain results briefly."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources folder.",
            "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": "Get descriptive statistics for a CSV column.",
            "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation and p-value between two loaded numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {"col1": {"type": "string"}, "col2": {"type": "string"}},
                "required": ["col1", "col2"],
            },
        },
    },
]
NODE_TOOLS = {
    **NODE_TOOLS,
    "load_csv": lambda args: csv_manager.load_csv(args["filename"]),
    "get_column_stats": lambda args: csv_manager.get_column_stats(args["column"]),
    "compute_correlation": lambda args: csv_manager.compute_correlation(args["col1"], args["col2"]),
}


def run_agent_cycle(messages, user_input: str, max_rounds: int = 8):
    """Append a user turn, run tools as requested, and append its final reply."""
    messages.append({"role": "user", "content": user_input})
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto"
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or ""
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            value = NODE_TOOLS[call.function.name](args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(value, default=str)})
    return "Stopped after reaching the tool-round limit."


# Q4-Q6 are run from assignments_07 so resources/bike_commute.csv resolves.
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print(result)
# Q5: the new correlation tool lets the agent complete the requested analysis.
# Q6: system gives the agent its instructions; user supplies the task; assistant
# reasons or requests a tool; tool records the result returned by our Python code.
print(json.dumps(messages, indent=2, default=str))


# --- Lesson 04: smolagents ---
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded bike commute data.

    Args:
        col1: Name of the first numeric DataFrame column.
        col2: Name of the second numeric DataFrame column.

    Returns:
        A dictionary with both column names, Pearson r, and p-value, or an error.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)
# smolagents derives a tool description and parameter schema from the function
# signature and docstring. The developer must explain purpose, arguments, and
# return meaning clearly; the manual JSON schema states those fields explicitly.

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(api_key=api_key, model_id=MODEL)


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources into the shared CsvManager.

    Args:
        filename: CSV file name located inside resources.

    Returns:
        The loaded table shape and column names, or an error.
    """
    return csv_manager.load_csv(filename)


@tool
def get_column_stats(column: str) -> dict:
    """Return descriptive statistics for a column in the loaded CSV.

    Args:
        column: Name of the column to summarize.

    Returns:
        Descriptive statistics or an error if unavailable.
    """
    return csv_manager.get_column_stats(column)


SMOL_TOOLS = [load_csv, get_column_stats, compute_correlation]
tool_agent = ToolCallingAgent(tools=SMOL_TOOLS, model=model, max_steps=8)
code_agent = CodeAgent(
    tools=SMOL_TOOLS,
    model=model,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("ToolCallingAgent:", response_tool)
print("CodeAgent:", response_code)
# Record what was observed when running: a ToolCallingAgent can only invoke the
# fixed tools and has no plot tool here, while the CodeAgent can generate a plot.
# A CodeAgent can tailor open-ended analyses; a ToolCallingAgent is more suitable
# when actions must stay within a small, explicitly defined set.


# --- Q9 Reflection ---
# A ToolCallingAgent is a better fit for a bounded task such as looking up an
# account balance through an approved API: the task maps to a known operation and
# the tool can enforce access checks and validation. A CodeAgent may generate and
# execute arbitrary code in its environment, so unsafe or incorrect generated
# code can read, overwrite, or expose data; a fixed ToolCallingAgent has no such
# general code execution capability.
