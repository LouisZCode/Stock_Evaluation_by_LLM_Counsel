

from langchain.agents import create_agent

minimax = create_agent(
    model="MiniMax-M2.1",
    system_prompt="you are a helpful comedian",
)

answer = minimax.invoke({"role" : "user"}, {"messages" : "Tell me a joke"})

print (answer)