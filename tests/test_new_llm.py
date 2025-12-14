"""
Here you can do different isolated LLM calls just to test the info is working and the API is sending info.
"""

import sys
sys.path.insert(0, "/Users/luiszg/Desktop/GitHub/Stock_Evaluation_by_LLM_Counsel")

from agents import mistral_finance_boy

response = mistral_finance_boy.invoke(
    {"messages" : [{"role": "user", "content": "Tell me more about palantir"}]},
    {"configurable" : {"thread_id" : "001_test"}}
)

for  i, msg in enumerate(response["messages"]):
    msg.pretty_print()