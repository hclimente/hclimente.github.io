---
layout: post
title: Agentic Workflows
date: 2025-10-12 11:59:00 +0000
description: Is this the future? Idk.
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

I have been a [Nextflow](https://www.nextflow.io/) evangelizer for close to 10 years. I have pioneered and promoted its use in numerous jobs, showing how it can facilitate computational tasks at any scale. Yet, there are some tasks that are impossible to tackle with Nextflow and other similar tools. Here's one:

> I can't be the only scientist that struggles to keep up with scientific papers. There are so many interesting articles published _every day_, I get into a decision paralysis of sorts, and I end up reading much less than I should. Hey, it's my rationalization; don't you dare popping my bubble! Wouldn't it be cool to have a pipeline that checks all my favorite journals daily, screens out all the papers I won't be interested in, and gives me the papers that I should read?

In this post, I will describe my journey creating an agentic workflow to parse and prioritize scientific articles.

# From Jupyter notebooks to agentic workflows

Classically, writing code implicitly creates a directed graph:

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/python_dag.webp"
    class="img-fluid mx-auto d-block"
    width="33%"
%}

This paradigm is common in data science scripts, and in most applications people use in their day to day.

With the advent of big data, computational needs grew. A lot. That's when modern orchestrators like [Airflow](https://airflow.apache.org/) or [nextflow](https://www.nextflow.io/) appeared. They not only make the computational graph explicit, but force the code to define a directed _acyclic_ graph (DAG), which ensures that there are no infinite loops:

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/nextflow_dag.webp"
    class="img-fluid mx-auto d-block"
    width="50%"
%}

This paradigm is common in large-scale computational workflows. Each process often has its own dependencies, and hence runs in its own container. It allows to easily parallelize and distribute the work.

Recently, the advent of LLMs has enabled **agentic workflows**, in which the programmer doesn't define a graph, only a set
of steps. However, it's not the programmer who decides how to combine them into the final workflow: it's an AI agent. The AI agent is given a goal and a set of tools, and is thrown into a job.

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/langchain_dag.webp"
    class="img-fluid mx-auto d-block"
    width="50%"
%}

As of October 2025, agents have issues with long contexts. A rule of thumb is to keep agentic workflows shorter than 10 steps or so. Since often our workflows will be much more complex than that, workflows will be hybrids that combine conventional workflows and (micro-)agentic workflows that take care of specific steps.

# Flavors of agentic workflow

An agentic workflow can be trivially expressed in a few lines of pseudocode:

```python
# The AGENT
llm = LLM(prompt="A prompt specifying a goal and a context")

# The CONTEXT
# a sequence of steps and outputs:
# step -> output -> step -> output -> ...
# provides the context to the llm
user_request = "The goal the user wants the agent to achieve"
context = [user_request]

# The LOOP
while True:
    # next_step = {
    #       "tool": tool to use next, or None if done
    #       "input": list of input(s) to the next step
    #   }
    next_step = decide_next_step(llm, context)

    if next_step["tool"] is None:
        break

    context.append(next_step)
    output = execute_step(next_step)
    context.append(output)

return context[-1]
```

The main components of an agentic workflow are:

- The **agent**: an LLM that will be making the decisions. The LLM has been initialized with a prompt that explains its goals and describes the available tools.
- The **context**: a list of the steps carried out so far and their outputs.
- The **loop**: an iterative approach that keeps asking the agent what to do next and executing it. Importantly, it needs a mechanism to determine when to exit the loop (e.g., when the agent's task is concluded).

Despite their relative simplicity, there are multiple frameworks that further simplify the process, like Google's [Agent Development Kit](https://google.github.io/adk-docs/), OpenAI's [Agent Builder](https://platform.openai.com/docs/guides/agent-builder) or LangChain's [LangGraph](https://www.langchain.com/langgraph). Because I am quite familiar with the Google ecosystem, I will focus on Google's Agent Development Kit (ADK) and use Gemini 2.5 Pro in AI Studio's [Free tier](https://ai.google.dev/gemini-api/docs/pricing).

Additional elements:

- Handoff: the action of an agent delegating a task to another agent.
- Guardrails: make sure that the agent stays on track.

## The tools

The tools are modular pieces of code that agents can call, like a function. Importantly, they must be well documented with a docstring that comprehensively describes that the tool can do and how to use it. Additionally, the ADK also comes with built-in tools. For instance, `google.adk.tools.google_search` allows the agent to search Google.

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
