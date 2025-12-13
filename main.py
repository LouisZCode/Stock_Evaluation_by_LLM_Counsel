from agents import groq_finance_boy, openai_finance_boy, anthropic_finance_boy


response = groq_finance_boy.invoke(
    {"messages" : "What can you do and what model are you?"},
    {"configurable" : {"thread_id" : "001"}}
)

for i, msg in enumerate(response["messages"]):
    msg.pretty_print()