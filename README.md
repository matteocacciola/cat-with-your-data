# Cat With Your Data

A Cheshire Cat AI plugin that lets you query your data in natural language.

It can reason over SQL databases, CSV files, and JSON files, then return a concise answer in chat.

<img src="./thumb.jpg" width="400" alt="Cat With Your Data thumbnail" />

## What It Does

- Turns user questions into datasource-aware reasoning steps.
- Uses dedicated LangChain agents for SQL, CSV, and JSON.
- Supports custom `input_prompt` and `output_prompt` templates.
- Works with multiple SQL engines through SQLAlchemy connection strings.

## Supported Datasources

From `settings.py`, the plugin supports:

- `PostgreSQL`
- `MySQL`
- `Oracle`
- `Microsoft SQL Server`
- `Microsoft SQL Server ODBC`
- `SQLite`
- `CSV`
- `JSON`

Reference for SQLAlchemy connection formats:

https://docs.sqlalchemy.org/en/20/core/engines.html

## Requirements

Install Python dependencies from `requirements.txt`:

- `langchain-experimental`
- `mysql-connector-python`
- `psycopg2-binary`
- `tabulate==0.9.0`

Depending on your datasource, additional DB drivers may be required by SQLAlchemy.

## Configuration

Plugin settings are defined in `settings.py` and loaded at runtime.

Main fields:

- `ds_type`: datasource type (for example `PostgreSQL`, `CSV`, `JSON`)
- `host`: DB host or local file path for CSV/JSON
- `port`: DB port
- `username`: DB username
- `password`: DB password
- `database`: DB name
- `input_prompt`: template used before reasoning (`{user_message}` available)
- `output_prompt`: final response template (`{prompt_prefix}`, `{user_message}`, `{thought}`, `{chat_history}` available)

Ready-to-use examples are in `settings_examples/`:

- `settings-postgres.json`
- `settings-mysql.json`
- `settings-csv.json`
- `settings-json.json`
- `settings-prompt.json`
- `settings-examples.json`

## How It Works

1. The hook `agent_fast_reply` (in `query_cat.py`) initializes `QueryCatAgent`.
2. `QueryCatAgent` (in `query_agent.py`) loads settings and picks the correct agent type.
3. The plugin runs datasource reasoning (`sql`, `csv`, or `json`) to build a thought.
4. The final answer is generated with the configured output prompt and chat context.

## Usage

1. Install the plugin in your Cheshire Cat environment.
2. Configure plugin settings (or start from one file in `settings_examples/`).
3. Ask questions in natural language, for example:
   - "How many products are in the catalog?"
   - "Show me total sales by month."
   - "Which item has the highest revenue?"

## Notes

- For `CSV` and `JSON`, set `host` to a readable file path.
- For SQL datasources, verify connectivity and credentials from the Cat runtime environment.
- If needed, tune `input_prompt` to inject query hints and business rules.
