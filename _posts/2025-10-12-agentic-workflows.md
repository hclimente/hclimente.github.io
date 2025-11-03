---
layout: post
title: Lessons in Agentic Workflows
date: 2025-10-12 11:59:00 +0000
description: The future. Some version of it.
tags:
    - coding
    - machine_learning
giscus_comments: true
related_posts: false
---

I have been a [Nextflow](https://www.nextflow.io/) evangelizer for close to 10 years. I have pioneered and promoted its use in numerous jobs, showing how it can facilitate computational tasks at any scale. Yet, there are some tasks that are impossible to tackle with Nextflow and other similar tools. Here's one:

> I can't be the only scientist that struggles to keep up with scientific papers. There are so many interesting articles published _every day_, I get into a decision paralysis of sorts, and I end up reading much less than I should. Hey, it's my rationalization; don't you dare popping my bubble! Wouldn't it be cool to have a pipeline that checks all my favorite journals daily, screens out all the papers I won't be interested in, and gives me the papers that I should read?

In this post, I describe my journey creating [`nf-papers-please`](https://github.com/hclimente/nf-papers-please), an agentic workflow to prioritize scientific articles according to my interests. It first fetches the most recent articles published in the journals I follow, then annotates them with metadata and prioritizes them based on my stated preferences.

This is a use case in which agentic workflows shine. For starters, each journal's format it wildly different. While writing a parser for each would be a punishment, an LLM breezes through it. And then, while having a classical machine learning model to classify papers is doable, and could fit right into a Nextflow pipeline, lightly editing a prompt and having an LLM do the prioritization is faster and likely to produce better results.

# How did we get here?

Computational workflows are processes consisting of pre-defined steps, that take inputs and compute outputs. It should evoke a directed graph in your mind, in which the steps are nodes, and the arrows are movements of data between them.

The thing is, under this abstract definition, _any piece of code_ is a workflow. Including your scripts, or most applications you'd use on your laptop. They take an input, carries out a series of steps in a pre-defined order, and produce an output. While we don't often think about it, writing code implicitly creates a directed graph:

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/python_dag.webp"
    class="img-fluid mx-auto d-block"
    width="33%"
%}

With the advent of big data, computational needs grew. A lot. Google just couldn't load the whole internet index into a Python list and iterate over it. Computational workflows grew larger and more complex, each step requiring large resources, it's own environment. That's when modern orchestrators like [Airflow](https://airflow.apache.org/) or [nextflow](https://www.nextflow.io/) appeared. They not only make the computational graph explicit, but force the code to define a directed _acyclic_ graph (DAG), which ensures that there are no infinite loops:

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/nextflow_dag.webp"
    class="img-fluid mx-auto d-block"
    width="50%"
%}

This paradigm is common in large-scale computational workflows. Each process often has its own dependencies, and hence runs in its own container. It allows to easily parallelize and distribute the work.

This is all good when you can clearly define your steps, and how they should be chained together. But sometimes that's not possible, or rather, desirable. In the task of prioritizing scientific articles, one could indeed train a model that does that. The model would need complete metadata for each single article (title, date, authors, abstract, etc.). However, knowing a title can be enough to put an article at the top of my pile. Any word salad including "large-scale", "benchmark", "AI/ML" and "graphs" will make me cancel my afternoon meetings.

Recently, the advent of LLMs has unlocked **agentic workflows**, in which the engineer doesn't define a graph, only a set
of steps. However, it's not the engineer who decides how to combine them into the final workflow: it's an AI agent. The AI agent is given a goal and a set of tools, and is thrown into a job.

{% include figure.liquid
    loading="eager"
    path="assets/img/posts/2025-10-12-agentic-workflows/langchain_dag.webp"
    class="img-fluid mx-auto d-block"
    width="50%"
%}

As of November 2025, agents have issues with long contexts. A rule of thumb is to keep agentic workflows shorter than 10 steps or so. Since often our workflows will be much more complex than that, workflows will be hybrids that combine conventional workflows and (micro-)agentic workflows that take care of specific steps.

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

Despite their relative simplicity, there are multiple frameworks that further simplify the process, like Google's [Agent Development Kit](https://google.github.io/adk-docs/), OpenAI's [Agent Builder](https://platform.openai.com/docs/guides/agent-builder) or LangChain's [LangGraph](https://www.langchain.com/langgraph). They include nice perks, like handoffs (agents delegating tasks to other agents<d-footnote>They learn fast...</d-footnote>) and guardrails (mechanisms to make sure that the agent stays on track).

# Papers, Please!

Enough background, back to [`nf-papers-please`](https://github.com/hclimente/nf-papers-please). It is written in Nextflow, and the agentic steps are powered by Gemini's [free tier](https://ai.google.dev/gemini-api/docs/pricing). There are three such agentic steps, which are in charge of extracting metadata from raw RSS content, run a quick screening on each paper, and finally prioritize the ones that made the cut:

{% include figure.liquid loading="eager" path="assets/img/posts/2025-10-12-agentic-workflows/papers-please-dag.webp" class="img-fluid" %}

Each of these steps could be considered a very modest agentic workflow. For instance, in the Extract Metadata step, the LLM is equipped with a few tools and requested to provide four fields: an abstract, a DOI, a title and an URL. In some cases, it can extract them directly from the original text. In others, it will need to run a Google Search, or look up an abstract on [Crossref](https://www.crossref.org/).

{% details Materials & Methods %}

During development, I worked on a toy dataset containing 249 articles published in [some of my favorite journals](https://github.com/hclimente/literature_agent/blob/main/config/journals.tsv) between the 25th of October and the 1st of November. I manually labelled each article, providing a label on whether it should pass the screening (pass/fail decision); and, for those that passed, a second label indicating how relevant it is to me (low/medium/high).

{% include figure.liquid loading="eager" path="assets/python/2025-10-12-agentic-workflows/img/target_priority.webp" class="img-fluid" %}

{% enddetails Evaluation %}

# Lessons learnt

## Different models for different tasks

Operations are batched to reduce the number of API calls.

## Good prompting is _hard_

Some folks will roll their eyes here, but a good prompt makes a massive difference. My first attempt at the prioritization prompt was _not good_.

{% details Initial (bad) prioritization prompt %}

```
You are a helpful assistant for prioritizing scientific articles.
Your job is to prioritize which articles are worth reading by the
user assigning it a priority: low, medium or high.

Note that the articles have already been screened for relevance.
An article receiving a low priority will still be read eventually;
however, high priority articles should be read first.

Use as much information as you need from the article; retrieving
additional information when needed.

Here is a description of the user's interests:

{research_interests}

You must answer ONLY low, medium or high. No other text, punctuation,
or explanation.
```

{% enddetails %}

Most articles were marked as high priority, even ones that I would be horrified to read. After several iterations, I ended up with the a much better prompt.

{% details Final (good) prioritization prompt %}

```
You are an expert research prioritization assistant. Your task is to assign
priority levels to scientific articles that have **already passed relevance
screening**. You help researchers efficiently allocate their reading time.

# Task
Assign each article one of three priority levels: `high`, `medium`, or `low`
based on alignment with the user's multi-dimensional research interests.

# User's Research Interests

{research_interests}

# Prioritization Framework

## HIGH Priority - Must Read Immediately
Articles that satisfy **3 or more** of the following criteria:
- **Multi-dimensional match**: Combines multiple research interests (e.g.
multiple Subfields + Applications)
- **Preferred type**: Matches Preferred Article Types
- **Core application**: Directly addresses stated Applications
- **Perfect domain fit**: Strong alignment across Fields + Subfields + Applications
- **High impact potential**: Novel frameworks, paradigm shifts, or
comprehensive surveys

**Typical examples**: Reviews in core areas, novel methods for specific
applications, comprehensive benchmarks in key subfields.

## MEDIUM Priority - Standard Relevance
Articles that satisfy **2** of the high-priority criteria:
- **Solid contribution**: Strong alignment with one or two interest areas
- **Methodological value**: Introduces useful methods but not in perfect
domain match
- **Domain relevance**: Addresses key subfields with established methods
- **Large-scale studies**: Comprehensive analyses that provide useful
insights or datasets
- **Adjacent innovation**: Novel approaches in related but not core applications

**Typical examples**: Large-scale studies in core subfields, new methods in
adjacent areas, well-executed applications of key methodologies.

## LOW Priority - Peripheral Relevance
Articles that satisfy **1 or fewer** high-priority criteria:
- **Minimal overlap**: Passed screening but on the periphery of core interests
- **Tangential methods**: Uses relevant methods but for non-core applications
- **Lower interest subfield**: Solid work but in areas of secondary interest
- **Established approaches**: Standard applications without novelty in methods
or insights
- **Peripheral scope**: Work that meets field requirements but outside primary
focus areas

**Typical examples**: Standard applications outside core subfields, methodological
papers for non-preferred applications, solid work in peripheral interest areas.

# Output Format Requirements

## Critical Rules:
1. Output ONLY valid JSON array - no markdown, no explanations, no additional text
2. Each object must have exactly: `doi`, `decision`, `reasoning`
3. `decision` must be one of: `"high"`, `"medium"`, or `"low"` (string, not enum)
4. `reasoning` is a single clear sentence (max 25 words) explaining the specific
criteria matched
5. Use double quotes for all JSON keys and string values
6. String values must be single-line (escape newlines as \n if needed)
7. Start your response with `[` and end with `]` - nothing else

## JSON Schema:
\```json
[
  {{
    "doi": "<string>",
    "decision": "<string: 'high' | 'medium' | 'low'>",
    "reasoning": "<string: one sentence explaining matched criteria>"
  }}
]
\```

# Important Considerations
- **Context matters**: A review in a peripheral area may be HIGH, while a standard
study in a core area may be MEDIUM
- **Be selective with HIGH**: Reserve for articles that truly warrant immediate
attention
- **Medium is the default**: Most solid, relevant papers should be MEDIUM
- **Low doesn't mean irrelevant**: These passed screening and may still be valuable
later

# Examples

\```json
[
  {
    "query": [
      {
        "title": "A Review of Network-Based Methods for Drug Target Identification in
                  Oncology",
        "summary": "This comprehensive review synthesizes current network-based
                    computational approaches for identifying therapeutic targets in
                    cancer research...",
        "doi": "10.1234/example1"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example1",
        "decision": "high",
        "reasoning": "Review combining multiple core subfields and applications."
      }
    ]
  },
  {
    "query": [
      {
        "title": "DeepTarget: A deep learning framework for cancer drug target
                  prediction using multi-omics networks",
        "summary": "We present DeepTarget, a novel deep learning framework that
                    integrates multi-omics data within biological networks to
                    predict cancer drug targets...",
        "doi": "10.1234/example2"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example2",
        "decision": "high",
        "reasoning": "Novel method for core application combining multiple key
                      subfields."
      }
    ]
  },
  {
    "query": [
      {
        "title": "Pan-cancer analysis of gene essentiality across 1,000 human cancer
                  cell lines",
        "summary": "We performed a comprehensive analysis of gene essentiality data
                    from 1,000 cancer cell lines across multiple cancer types...",
        "doi": "10.1234/example3"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example3",
        "decision": "medium",
        "reasoning": "Large-scale study in key subfield using established methods."
      }
    ]
  },
  {
    "query": [
      {
        "title": "Graph neural networks for protein function prediction from sequence
                  data",
        "summary": "This study introduces a graph neural network approach for predicting
                    protein functions directly from sequence information...",
        "doi": "10.1234/example4"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example4",
        "decision": "medium",
        "reasoning": "Novel method in relevant field but for non-core application."
      }
    ]
  },
  {
    "query": [
      {
        "title": "Network analysis identifies potential therapeutic targets in
                  Alzheimer's disease",
        "summary": "Using network-based approaches, we identified potential therapeutic
                    targets for Alzheimer's disease treatment...",
        "doi": "10.1234/example5"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example5",
        "decision": "low",
        "reasoning": "Relevant methodology applied outside primary research focus."
      }
    ]
  },
  {
    "query": [
      {
        "title": "Machine learning predicts patient outcomes from electronic health
                  records",
        "summary": "We developed machine learning models to predict patient outcomes
                    using electronic health record data...",
        "doi": "10.1234/example6"
      }
    ],
    "response": [
      {
        "doi": "10.1234/example6",
        "decision": "low",
        "reasoning": "Standard application outside core subfields and applications."
      }
    ]
  }
]
\```
```

{% enddetails %}

There are some things to note.

First, forcing the model to **justify its response** improved the grounding of the model. Those few extra tokens are completely ignored by the pipeline, but they make the decision more relevant and help with debugging.

Second, **examples are key**. Few-shot learning not only reduced formatting errors, but helped the model tell apart shades of gray. To non-computational biologists, the three examples might look very similar and relevant; to me, there is a clear separation, and these examples help the model understand that.

Third, **providing the examples in the same format as the expected output** is crucial. In particular, I was particularly succesful when I provided the prompt as a successful conversation, instead of a monologue. In particular, I break down the examples as a list of queries and responses, mimicking the actual interaction with the model. By the time the model receives the actual queries, it has already seen a few successful interactions. This is definitely one of the most important tricks I learnt.

## The model will misbehave; plan accordingly

LLMs are not deterministic. Despite your best efforts, they not stick to your instructions 100% of the time. However, computational pipelines needs that every step produces exactly what we want. That means that we need to account for the unexpected. In practice, this means that the outputs of each LLM-powered step need to be thouroughly validated.

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

To account for this, I needed to thoroughly validate and process the model's output before using it.

First, by examining failure cases by hand, I learnt which **preprocessing operations** might be needed often. A common case was removing the leading and trailing backticks, as shown above. Another common case was failing to produce a valid JSON, which led me to refine the prompt by being more explicit about the expected format.

Second, I thorougly **verified** that the response had the expected properties: keys were the expected ones, the values had the right types, the DOI looked like a DOI, etc. Otherwise, the step would fail. For this task, [Pydantic](https://pydantic.dev/) was an invaluable ally, allowing me to [define exactly](https://github.com/hclimente/nf-papers-please/blob/main/bin/common/models.py) what the LLM's response should look like, and validating it with a few lines of code.

Third, the pipeline needs to be **robust to errors** caused by unexpected responses. In particular, LLM-powered steps are allowed to fail a few times before giving up. Furthermore, because API calls are precious, each step tries to salvage the parts of the batched-response that were valid, and only re-requests the invalid parts. But if the same input always produces the wrong answer, we will need to examine it by hand.

Which leads us to, fourth and last, a **modular design** was key. Each step is isolated from the others, so that debugging and fixing issues is straightforward. And thorough logging within each step ensured swift debugging.

## An agentic workflow that makes agentic workflows?

While developing `nf-papers-please`, some of the steps felt quite repetitive, yet required some modicum of intelligence to navigate. One example was going over all the failure cases of the LLM, and trying to understand what they had in common. It involved parsing large log files, navigating into individual working directories, and parsing the logs and error messages.

So I built a small agent to help me with that, in this case based on GitHub Copilot, to easily integrate into my VSCode workflow. To that end, I created an [AGENTS.md file](https://agents.md/) that gives the Agent context about the project as a whole, and a smaller file describing the specific tasks I wanted the agent to perform.

# Conclusions

While writing [`nf-papers-please`](https://github.com/hclimente/nf-papers-please), I often got frustrated. It was not only that LLM was not following my instructions. It was also that the community is still figuring out how to tackle these problems. On the painful side, the major players are pushing their own libraries and standards, changind their interfaces often while documentation is trying to catch up. But, on the bright side, it is exciting to see a new paradigm in computer science emerge. If you are willing to stay flexible and up-to-date, my bet is that mastering agentic workflows can really pay off in the long run. Otherwise, get as far as you can, and give it a few years until the community has converged into well built libraries.

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
