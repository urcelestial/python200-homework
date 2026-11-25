"""Conversational analysis of the World Happiness dataset with smolagents."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

df = None
DATA_DIR = Path("happiness_project")
OUTPUT_DIR = Path("outputs")

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load and merge all yearly World Happiness CSV files into memory.

    Reads files named world_happiness_YEAR.csv from happiness_project/. The
    source uses semicolon separators and decimal commas; the function standardizes
    country, region, happiness_score, gdp_per_capita, and year column names.

    Returns:
        Dataset shape and column names, or an error if no yearly files are found.
    """
    global df
    paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    if not paths:
        return {"error": f"No world_happiness_*.csv files found in {DATA_DIR.resolve()}"}
    frames = []
    for path in paths:
        try:
            year_df = pd.read_csv(path, sep=";", decimal=",")
            year_df = year_df.rename(columns=COLUMN_NAMES)
            year_df["year"] = int(path.stem.split("_")[-1])
        except (ValueError, OSError, pd.errors.ParserError) as error:
            return {"error": f"Could not read {path.name}: {error}"}
        frames.append(year_df)
    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": list(df.columns)}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single loaded dataset column.

    Args:
        column: Exact name of the column to describe.

    Returns:
        The pandas descriptive statistics as a dictionary, or an error if data
        has not been loaded or the column does not exist.
    """
    if df is None:
        return {"error": "Load the happiness data before summarizing a column."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson's correlation and p-value between numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing col1, col2, pearson_r, and p_value, rounded to
        four decimal places, or an error when input data is invalid.
    """
    if df is None:
        return {"error": "Load the happiness data before computing a correlation."}
    missing = [column for column in (col1, col2) if column not in df.columns]
    if missing:
        return {"error": f"Column not found: {', '.join(missing)}"}
    try:
        pair = df[[col1, col2]].dropna()
        if not pd.api.types.is_numeric_dtype(pair[col1]) or not pd.api.types.is_numeric_dtype(pair[col2]):
            return {"error": "Both columns must contain numeric data."}
        r_value, p_value = pearsonr(pair[col1], pair[col2])
    except (TypeError, ValueError) as error:
        return {"error": str(error)}
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r_value), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries by a selected column for a particular year.

    Args:
        column: Numeric column used to rank countries, such as happiness_score.
        year: Year to filter before ranking.
        n: Number of highest-ranking rows to return; defaults to five.

    Returns:
        A list of dictionaries with country and the selected value, or an error
        for missing data, columns, years, or invalid counts.
    """
    if df is None:
        return {"error": "Load the happiness data before ranking countries."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    if "year" not in df.columns or "country" not in df.columns:
        return {"error": "Dataset must contain year and country columns."}
    if not isinstance(n, int) or n < 1:
        return {"error": "n must be a positive integer."}
    selected = df.loc[df["year"] == year, ["country", column]].dropna()
    if selected.empty:
        return {"error": f"No rows found for year {year}."}
    if not pd.api.types.is_numeric_dtype(selected[column]):
        return {"error": f"Column is not numeric: {column}"}
    return selected.sort_values(column, ascending=False).head(n).to_dict(orient="records")


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file before running the project.")

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
SYSTEM_PROMPT = """You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses."""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

QUERIES = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


def main():
    """Run the guided queries and student-selected follow-up questions."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not DATA_DIR.is_dir() or not list(DATA_DIR.glob("world_happiness_*.csv")):
        raise FileNotFoundError(
            f"Copy the yearly World Happiness CSV files into {DATA_DIR.resolve()} before running."
        )
    for query in QUERIES:
        print(f"\n--- Query: {query} ---")
        print(agent.run(query, reset=False))
    plot_path = OUTPUT_DIR / "happiness_by_region.png"
    if not plot_path.is_file():
        raise RuntimeError(f"The plotting query completed without creating {plot_path}.")

    # This compares the average happiness score by region in 2015 and 2024.
    my_query_1 = (
        "Using Python code, compare each region's average happiness_score in 2015 and 2024. "
        "Show the change in a sorted horizontal bar chart and save it to outputs/region_change.png."
    )
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # This should use code to group, compare, and plot because no tool covers that analysis.

    my_query_2 = "Use the tools to list the top 3 countries by gdp_per_capita in 2020."
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # This should call get_top_n_countries, with no custom code needed.


if __name__ == "__main__":
    main()


# --- Reflection ---
# 1. After running Query 3, record the agent's reported r, p-value, and whether
#    it explains both the direction/strength and statistical significance.
# 2. After running the guided queries, record one specific surprising response;
#    do not infer the result before observing the model's run.
# 3. A useful additional tool would calculate changes in a measure between two
#    selected years for each region, returning a ranked table. It would answer
#    questions about which regions improved or declined most without asking the
#    CodeAgent to write grouping and comparison code each time.
