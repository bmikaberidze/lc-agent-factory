"""
Minimal example: create and run an agent with LCAgentFactory.

LCAgentFactory.create() accepts a plain dict — load it however you like:

    import yaml, json, os
    config = yaml.safe_load(open("config.yaml"))   # YAML file
    config = json.load(open("config.json"))         # JSON file
    config = json.loads(os.environ["AGENT_CONFIG"]) # env var
    config = {"models": {"primary": {...}}}         # inline

Prerequisites:
    pip install lc-agent-factory langchain-google-genai pyyaml
    export GOOGLE_API_KEY="..."
"""

import yaml
config = yaml.safe_load(open("examples/config.yaml"))

from lc_agent_factory import create_agent
agent = create_agent(config)

res = agent.invoke({"messages": [
    {"role": "user", "content": "Hello, are you the agent I created?"},
]})
print(res["messages"][-1].content)
