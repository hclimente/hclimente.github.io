---
layout: post
title: Agentic workflows
date: 2025-10-12 11:59:00 +0000
description: The secret sauce of machine learning
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

I can't be the only scientist that struggles to keep up with scientific papers. There are so many interesting articles published _every day_, I get into a decision paralysis of sorts, and I end up reading much less than I should. Hey, it's my rationalization; don't you dare popping my bubble! Surely, the solution this is a problem I can solve with AI, right? Surely once I have

In this post, I will describe my journey creating an agentic workflow to parse and prioritize scientific articles.

# From computational workflows to agentic workflows

Classically, writing code implicitly creates a directed graph. Modern orchestrators like [Airflow](https://airflow.apache.org/) or [nextflow](https://www.nextflow.io/) not only make this explicit, but force the code to define a directed _acyclic_ graph (DAG), which ensures that there are no infinite loops.

Agentic workflows still rely on deterministic, pre-defined steps. However, it's not the human who decides how to combine them into the final workflow: it's an AI agent. The AI agent is given a goal and a set of tools, and is thrown into a job. If I were to summarize it into a piece of pseudocode:

```python
user_input = """
A prompt specifying a goal and a context
"""

# a sequence of steps and outputs
# provides the context to the agent
context = [user_input]

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

Additional elements:

- Handoff: the action of an agent delegating a task to another agent.
- Guardrails: make sure that the agent stays on track.

## Tools

Many frameworks come already with tools (e.g., for websearch) and allow to create new tools.

# Agentic workflows

While an agentic workflow doesn't need more than 9 lines of code, there are multiple frameworks that further simplify the process. In this post, I will focus on Google's [Agent Development Kit](https://google.github.io/adk-docs/), using Gemini 2.5 Pro in AI Studio's [Free tier](https://ai.google.dev/gemini-api/docs/pricing).

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
