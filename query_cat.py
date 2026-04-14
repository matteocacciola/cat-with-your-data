from cat import hook, AgenticWorkflowOutput

from .query_agent import QueryCatAgent

@hook
async def agent_fast_reply(cat) -> AgenticWorkflowOutput | None:
    # Instantiate query agent
    query_agent = QueryCatAgent(cat)

    # Get thought from a reasoning agent
    thought = await query_agent.get_reasoning_agent()
    if not thought:
        return None
    
    # Get a final and contextual response
    return await query_agent.get_final_output(thought)
