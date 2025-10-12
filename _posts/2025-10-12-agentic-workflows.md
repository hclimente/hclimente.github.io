---
layout: post
title: Agentic workflows
date: 2025-08-02 11:59:00 +0000
description: The secret sauce of machine learning
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

In a very in-fashion call, I will build a small workflow to optimize how I prioritize literature articles.

# Background: agentic workflows

Classically, writing code implicitly creates a directed graph. Modern orchestrators like [Airflow](https://airflow.apache.org/) or [nextflow](https://www.nextflow.io/) not only make this explicit, but force the code to define a directed _acyclic_ graph (DAG), which ensures that there are no infinite loops. Agentic workflows still rely on deterministic, pre-defined steps. However, it's not the human who decides how to combine them into the final workflow: it's an AI agent. The AI agent is given a goal and a set of tools, and is thrown into a job. If I were to summarize it into a piece of pseudocode:

```python
# a sequence of steps and outputs
# provides the context to the agent
context = [initial_event]

while True:
    # next_step = {
    #       "tool": tool to use next, or None if done
    #       "input": list of input(s) to the next step
    #   }
    next_step = agent.decide_next_step(context)

    if next_step["tool"] is None:
        break

    context.append(next_step)
    output = execute_step(next_step)
    context.append(output)

return context[-1]
```

> As of August 2025 agents have issues with long contexts. Hence, reliable workflows should be achievable in 3 to 10 steps. They take care of specific steps within workflows that are surrounded by deterministic code.

As the code shows, the main components of an agentic workflow:

- Agent: an LLM that will be making the decisions
- Prompt: a text explaining the agent's goals and describing the available goals
- Context: a list of steps so far and their outputs
- Loop: an iterative approach that keeps getting asking the agent what tool to use next. Importantly, it needs a mechanism to determine when to exit the loop (e.g., when the agent's task is concluded).

## Tools

We have gloss

# Pre-requisites: the free version

I started out playing with open source & open weights components, namely [Ollama](https://ollama.ai/) and its [Langchain integration](https://python.langchain.com/docs/integrations/chat/ollama/).

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
