---
layout: post
title: Lessons in Agentic Workflows
date: 2025-10-12 11:59:00 +0000
description: Is this the future? Idk.
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

I have been a [Nextflow](https://www.nextflow.io/) evangelizer for close to 10 years. I have pioneered and promoted its use in numerous jobs, showing how it can facilitate computational tasks at any scale. Yet, there are some tasks that are impossible to tackle with Nextflow and other similar tools. Here's one:

> I can't be the only scientist that struggles to keep up with scientific papers. There are so many interesting articles published _every day_, I get into a decision paralysis of sorts, and I end up reading much less than I should. Hey, it's my rationalization; don't you dare popping my bubble! Wouldn't it be cool to have a pipeline that checks all my favorite journals daily, screens out all the papers I won't be interested in, and gives me the papers that I should read?

In this post, I will describe my journey creating an agentic workflow to parse and prioritize scientific articles. The name of our agent? Vör 🧙‍♀️.

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

Despite their relative simplicity, there are multiple frameworks that further simplify the process, like Google's [Agent Development Kit](https://google.github.io/adk-docs/), OpenAI's [Agent Builder](https://platform.openai.com/docs/guides/agent-builder) or LangChain's [LangGraph](https://www.langchain.com/langgraph).

Additional elements:

- Handoff: the action of an agent delegating a task to another agent.
- Guardrails: make sure that the agent stays on track.

# Our workflow

Enough background. Let's get hands on.

I will be developing a pipeline to retrieve and prioritize scientific articles. The pipeline will fetch the most recent articles published in the journals I follow, annotate them with metadata and prioritize them based on my stated preferences. This is a use case in which agentic workflows should shine. Each journal publishes their articles in a very different format. While writing a parser for each editorial would be a punishment, an LLM equipped with a few tools should breeze through it. Then, assessing the relevance to me of an article should just be a matter of matching that metadata to my preferences.

For development, I will work on a toy dataset containing 267 entries retrieved from [some of the journals I follow](https://github.com/hclimente/literature_agent/blob/main/config/journals.tsv). I will focus on the Gemini family of models tying to make the most of their [Free tier](https://ai.google.dev/gemini-api/docs/pricing) while getting acceptable results.

Here is the design of the pipeline I pursued:

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/vor_dag.webp"
    class="img-fluid mx-auto d-block"
    width="50%"
%}

# Lessons learnt

## Different models for different tasks

## Good prompting is _hard_

Some folks will roll their eyes here, but a good prompt makes a massive difference. Here is an example of my first attempt at prompting:

```
You are a helpful assistant for prioritizing scientific articles. Your job is to prioritize which articles
are worth reading by the user assigning it a priority: low, medium or high.

Note that the articles have already been screened for relevance. An article receiving a low priority will still be read eventually; however, high priority articles should be read first.

Use as much information as you need from the article; retrieving additional information when needed.

Here is a description of the user's interests:

{research_interests}

You must answer ONLY low, medium or high. No other text, punctuation, or explanation.
```

The results were just not good. Most articles were marked as high priority, even ones that I would be horrified to read. This is the current one:

```
You are an expert AI assistant for prioritizing scientific articles for a researcher. Your job is to assign a priority level (`high`, `medium`, or `low`) to articles that have **already been screened for general relevance**.

Your output MUST be a single, minified JSON object with NO additional text, explanation, or markdown formatting.

# User's Research Interests:

{research_interests}

# Prioritization Rubric:
You will assign priority based on how many of the user's interest dimensions the article satisfies.

*   **high:** A must-read. These articles directly combine multiple high-priority interests. They are typically **Reviews** or **New Computational Methods** that fall squarely within the user's **Core Applications** and **Key Subfields** (e.g., a new Network Biology method for drug discovery in cancer).
*   **medium:** A standard, relevant article. These are solid contributions that strongly align with one or two interest areas but aren't a perfect multi-point match. This could be a **Large-Scale Analysis** in a key subfield or a new method that is slightly peripheral to the core applications.
*   **low:** Relevant, but can wait. These articles passed the initial screen but are on the periphery of the user's core focus. They might use established methods to study a specific disease that is not cancer, or focus on a subfield of lesser interest.

# JSON Output Structure:

{{
  "decision": "<string, one of: 'high', 'medium', or 'low'>",
  "reasoning": "<string, a brief one-sentence explanation for the assigned priority>"
}}

# Example 1: High Priority
**Article Title:** "A Review of Network-Based Methods for Drug Target Identification in Oncology"
**Correct Output:**
{{"decision":"high","reasoning":"This is a review article that perfectly combines three core interests: Network Biology, Drug Target Discovery, and Cancer Biology."}}

# Example 2: Medium Priority
**Article Title:** "A large-scale benchmark of machine learning models for predicting gene essentiality in 1,000 human cancer cell lines"
**Correct Output:**
{{"decision":"medium","reasoning":"This is a preferred article type (benchmark, large-scale analysis) in a key subfield (Cancer Biology), but does not introduce a new method or focus on drug discovery."}}

# Example 3: Low Priority
**Article Title:** "Application of gene co-expression networks to identify candidate genes for Alzheimer's disease"
**Correct Output:**
{{"decision":"low","reasoning":"While it uses a relevant method (Network Biology), its application is on a disease outside the user's core focus on cancer."}}
```

There are some things to note. First, adding a **reasoning** to the response improves the grounding of the model. Those few extra tokens are completely ignored by the pipeline, but they make the decision more relevant and help with debugging. Second, **examples are key**. They not only reduce formatting errors, but help the model tell apart shades of gray. To non-computational biologists, the three examples might look very similar and relevant; to me, there is a clear separation, and these examples help the model understand that.

## The model will misbehave; plan accordingly

LLMs are not deterministic. Despite your best efforts, they not always will stick to your prompts; in particular, when using relatively smaller models. However, computational pipelines needs that every step produces exactly what we want. That means that we need to account for the unexpected. In practice, this means that the outputs of each LLM-powered step need to be thouroughly validated.

A perfect example comes from the metadata extractor. The model here is expected to produce a JSON-formatted string containing three pieces of metadata:

```json
{
    "title": "...",
    "summary": "...",
    "doi": "..."
}
```

However, the model won't always stick to this format. A common departure involves wrapping the outcome in Markdown notes:

```
\```json
{
    "title": "...",
    "summary": "...",
    "doi": "..."
}
\```
```

During training, the model got used to see these two together. Old habits die hard.

To account for this, I needed to thoroughly validate and process the model's output before using it. First, by examining failure cases by hand, I learnt which **preprocessing operations** might be needed often. A common case was removing leading and trailing backticks. Second, I thorougly **verified** that the response had the expected properties: keys were the expected ones, the values had the right types, the DOI looked like a DOI, etc. Otherwise, the step would fail. And third, the pipeline would be **robust to errors**. In particular, when the response verification fails, we will simply roll the dice again and cross out fingers that this time the model behaves. But if the same input always produces the wrong answer, we will need to examine it by hand. Which leads us to, forth and last, **thoroughly logging** each step, to ensure swift debugging.

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
