import json
from pathlib import Path

from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent, JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.utilities import SQLDatabase
from langchain_experimental.agents import create_csv_agent
from cat import StrayCat, AgenticWorkflowTask, AgenticWorkflowOutput
from cat.log import log
from cat.templates import prompts

from .settings import datasources


class QueryCatAgent:
    def __init__(self, cat: StrayCat) -> None:
        self.cat = cat
        self.large_language_model = None
        self.agentic_workflow = None
        self.settings = None

    @classmethod
    async def create(cls, cat: StrayCat):
        instance = cls(cat)
        instance.large_language_model = await cat.large_language_model()
        instance.agentic_workflow = await cat.agentic_workflow()
        return instance

    # Load configurations
    async def _load_configurations(self):
        # Acquire settings
        settings = await self.cat.mad_hatter.get_plugin().load_settings()

        # If the settings are the same, skip the function
        if self.settings and self.settings == settings:
            return

        log.critical("Load query examples..")

        # Set settings
        self.settings = settings

    def _get_input_prompt(self) -> str:
        # Get user message
        user_message = self.cat.working_memory.user_message.text

        # Get input prompt from settings
        input_prompt = user_message
        if self.settings["input_prompt"] != '':
            input_prompt = self.settings["input_prompt"].format(
                user_message=user_message
            )

        log.debug("=====================================================")
        log.debug(f"Input prompt:\n{input_prompt}")
        log.debug("=====================================================")

        return input_prompt

    # Execute agent to get a final thought, based on the type 
    async def get_reasoning_agent(self) -> str | None:
        # Load configurations
        await self._load_configurations()

        # Acquire the agent type
        datasource_type = self.settings["ds_type"]
        agent_type = datasources[datasource_type]["agent_type"]

        # Execute agent based on the type
        if agent_type == "sql":
            return await self._get_reasoning_sql_agent()
        if agent_type == "csv":
            return await self._get_reasoning_csv_agent()
        if agent_type == "json":
            return await self._get_reasoning_json_agent()

        return None

    # Return the final response, based on the user's message and reasoning
    async def get_final_output(self, thought: str) -> AgenticWorkflowOutput:
        user_message = self.cat.working_memory.user_message.text

        # Load configurations
        await self._load_configurations()

        # Get prompt
        prompt_prefix = await self.cat.mad_hatter.execute_hook(
            "agent_prompt_prefix", prompts.MAIN_PROMPT, caller=self.cat
        )

        # Get user message and chat history
        chat_history = [history.content.text for history in self.cat.working_memory.history]

        # Default output Prompt
        output_prompt = f"""{prompt_prefix}
You have elaborated the user's question, you have searched for the answer and now you have the solution in your Thought; 
reply to the user briefly, precisely and based on the context of the dialogue.
- Human: {user_message}
- Thought: {thought}
- AI:"""

        # Set output prompt from settings
        if self.settings["output_prompt"]:
            output_prompt = self.settings["output_prompt"].format(
                prompt_prefix=prompt_prefix,
                user_message=user_message,
                thought=thought,
                chat_history=chat_history
            )

        # Invoke LLM and get a final and contextual response
        log.debug("=====================================================")
        log.debug(f"Output prompt:\n{output_prompt}")
        log.debug("=====================================================")

        agent_input = AgenticWorkflowTask(system_prompt=output_prompt, user_prompt=user_message)
        callbacks = await self.cat.plugin_manager.execute_hook("llm_callbacks", [], caller=self.cat)
        return await self.agentic_workflow.run(
            task=agent_input,
            llm=self.large_language_model,
            callbacks=callbacks,
        )

    async def _execute(self, agent_executor) -> str | None:
        # Get final thought, after agent reasoning steps
        try:
            final_thought = await agent_executor.ainvoke(self._get_input_prompt())
            return getattr(final_thought, "output", getattr(final_thought, "content", str(final_thought)))
        except Exception as e:
            log.error(f"Failed to execute the agent: {e}")
            return None

    # Execute sql agent
    async def _get_reasoning_sql_agent(self) -> str | None:
        # Create a connection string
        datasource_type = self.settings["ds_type"]
        connection_string = datasources[datasource_type]["conn_str"].format(**self.settings)
        log.info(f"Connection string: {connection_string}")

        # Create a SQL connection
        try:
            db = SQLDatabase.from_uri(connection_string)

            # Create SQL Agent
            agent_executor = create_sql_agent(
                llm=self.large_language_model,
                toolkit=SQLDatabaseToolkit(db=db, llm=self.large_language_model),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
        except Exception as e:
            log.error(f"Failed to create SQL connection: {e}")
            return None

        return await self._execute(agent_executor)

    # Execute csv agent
    async def _get_reasoning_csv_agent(self) -> str | None:
        # Get csv file path
        csv_file_path = self.settings["host"]

        # Create a CSV agent
        try:
            agent_executor = create_csv_agent(self.large_language_model, csv_file_path, verbose=True)
        except Exception as e:
            log.error(f"Failed to create SQL connection: {e}")
            return None

        return await self._execute(agent_executor)

    # Execute json agent
    async def _get_reasoning_json_agent(self) -> str | None:
        # Get the JSON file path
        json_file_path = self.settings["host"]

        # Create JSON agent
        try:
            # Get json data
            data = json.loads(Path(json_file_path).read_text(encoding="utf-8"))

            # Create JSON toolkit
            json_spec = JsonSpec(dict_=data, max_value_length=4000)
            json_toolkit = JsonToolkit(spec=json_spec)

            agent_executor = create_json_agent(
                llm=self.large_language_model,
                toolkit=json_toolkit,
                verbose=True
            )
        except Exception as e:
            log.error(f"Failed to create SQL connection: {e}")
            return None

        return await self._execute(agent_executor)
