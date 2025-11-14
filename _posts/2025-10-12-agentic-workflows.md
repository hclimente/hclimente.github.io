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

I can't be the only scientist that struggles to keep up with scientific papers. There are so many interesting articles published _every day_, I get into a decision paralysis of sorts, and I end up reading much less than I should. Hey, it's my rationalization; don't you dare popping my bubble! Wouldn't it be cool to have a pipeline that checks all my favorite journals daily, screens out all the papers I won't be interested in, and gives me the papers that I should read?

In this post, I describe my journey creating [`nf-papers-please`](https://github.com/hclimente/nf-papers-please), an agentic workflow to prioritize scientific articles according to my interests. It fetches the most recent articles published in my favorite journals, annotates them with metadata and prioritizes them based on my reading preferences.

This is a use case in which agentic workflows should shine. For starters, each journal's format it wildly different. While writing a parser for each would be a punishment, an LLM breezes through it. And then, while having a classical machine learning model to classify papers is doable, and could fit right into a Nextflow pipeline, lightly editing a prompt and having an LLM do the prioritization is faster and likely to produce better results.

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

Enough background, back to [`nf-papers-please`](https://github.com/hclimente/nf-papers-please). It is written in Nextflow, and the agentic steps are powered by Gemini's [free tier](https://ai.google.dev/gemini-api/docs/pricing).

{% details Materials & Methods %}

During development, I worked on a toy dataset containing 249 articles published in [some of my favorite journals](https://github.com/hclimente/literature_agent/blob/main/config/journals.tsv) between the 25th of October and the 1st of November. I manually labelled each article, providing a label on whether it should pass the screening (pass/fail decision); and, for those that passed, a second label indicating how relevant it is to me (low/medium/high).

<!--TODO show train & test distribution-->

{% include figure.liquid loading="eager" path="assets/python/2025-10-12-agentic-workflows/img/target_priority.webp" class="img-fluid" %}

I then randomly split the dataset into a training set (200 articles) and a test set (49 articles). During development, I only used the training set to refine my prompts, the pipeline and generate examples for few-shot learning. Once I was satisfied with the results, I ran the final pipeline on the test set to evaluate its performance. Unless otherwise noted, all results reported in this post correspond to the test set.

Unless otherwise noted, all LLM calls were made to [Gemini 2.5 Flash Lite](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite).

{% enddetails Evaluation %}

## v0.1: The naive take

In my [first approach](https://github.com/hclimente/nf-papers-please/tree/v0.1-beta), I encoded my manual curation process into a workflow with three agentic steps. They repectively extract metadata from raw RSS content, screen each paper for broad relevance, and finally prioritize the ones that made the cut:

{% include figure.liquid loading="eager" path="assets/img/posts/2025-10-12-agentic-workflows/papers-please-dag.webp" class="img-fluid" %}

Each of these steps is a very modest agentic workflow. For instance, in the Extract Metadata step, the LLM is equipped with a few tools and requested to provide four fields: an abstract, a DOI, a title and an URL. In some cases, it can extract them directly from the original text. In others, it will need to run a Google Search, or look up an abstract on [Crossref](https://www.crossref.org/).

<!--TODO show confusion matrix-->

## v0.2: the scoring system

On my [second approach](https://github.com/hclimente/nf-papers-please/tree/v0.2-beta) I tried to encode my preferences as a scoring system. I would [assign a positive or negative weight](https://github.com/hclimente/nf-papers-please/blob/v0.2-beta/config/research_interests.md) to the different dimensions I cared about, and have the LLM detect which ones applied to each article. For instance, if an article was published in Nature Genetics, it would get a +2 boost. If it was a preprint, it would get a soft -1 penalty. If it focused on network biology, it would get a +3 boost. And so on.

I got two learnings from this exercise: the LLM is terrible at basic arithmetic, and interactions matter a lot.

<!--TODO show boxplot-->

## v0.3: the RAG

Encoding my preferences into a prompt, even with points proved to be quite hard. So, in my [third attempt](https://github.com/hclimente/nf-papers-please/tree/v0.3-beta) I gave the LLM access to the articles I have actually read to guide its decisions. To that end, I created a small [retrieval-augmented generation (RAG) system]({% post_url 2025-08-16-rags %}) to retreive the most similar articles to the one being evaluated, and provide them as context to the LLM. This should help the model calibrate its decision, by comparing how similar the retrieved articles were among themselves and to the target article.

<!--TODO show confusion matrix-->

# What went wrong?

I believe my approach had two major flaws. For one, being limited to free tiers severely limits the quality of my results. While Gemini 2.5 Pro rocks the #1 position at [LMArena](https://lmarena.ai/leaderboard/text), Gemini 2.5 Flash Lite is at #50. This matters. When I presented bad decisions made by Flash Lite to Pro, Pro usually agreed that they were bad decisions.

Another problems come from the lack of supervised data. The model only has my prompt, or some positive examples. But not a set of negative examples. In particular, not negative examples that are _close_ to the positive ones. This makes it hard for the model to learn the fine distinctions that matter to me.

Last, the model can't capture my _evolving_ interests. An interesting conversation with a colleague might make me want to read more about a topic I'd have ignored before. Also, my Zotero library goes back nearly a decade, adding noise to the RAG step.

# What did I learn?

## Different models for different tasks

Operations are batched to reduce the number of API calls.

## Good prompting is _hard_

Some folks will roll their eyes here, but a good prompt makes a massive difference. My first attempt at the prioritization prompt was _not good_:

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

Most articles were marked as high priority, even ones that I would be horrified to read. After several iterations, I ended up with the a [much longer prompt](https://github.com/hclimente/nf-papers-please/blob/992bb02d48efd2fb4653549ff0ef12061d2d8d8f/prompts/prioritization.md) that produced more nuanced results.

There are some things to note.

First, forcing the model to **justify its response** improved the grounding of the model. Those few extra tokens are completely ignored by the pipeline, but they make the decision more relevant and help with debugging.

Second, **examples are key**. Few-shot learning not only reduced formatting errors, but helped the model tell apart shades of gray. To non-computational biologists, the three examples might look very similar and relevant; to me, there is a clear separation, and these examples help the model understand that.

Third, **providing the examples in the same format as the expected output** is crucial. In particular, I was particularly succesful when I provided the prompt as a successful conversation, instead of a monologue. In particular, I break down the examples as a list of queries and responses, mimicking the actual interaction with the model. By the time the model receives the actual queries, it has already seen a few successful interactions. This is definitely one of the most important tricks I learnt.

## Stay objective

The above prompt produced decent, but uncalibrated results. The LLM was just moderately amazed by any paper in Computational Biology (one of my stated fields of interests), and equally boosted them all, regardless of what they were about. I didn't find this too surprising. If you are not too into the weeds of _my_ flavour of Computational Biology, the safest bet is to give it a weak recommendation.

To help the model stay calibrated, I replaced that prompt by a scoring system. This had two goals: giving an univocal quantity representing how much I care about each property; and replacing hard categories by a semi-continuous score.

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

First, I thorougly **verified** that the response had the expected properties: keys were the expected ones, the values had the right types, the DOI looked like a DOI, etc. Otherwise, the step would fail. For this task, [Pydantic](https://pydantic.dev/) was an invaluable ally, allowing me to [define exactly](https://github.com/hclimente/nf-papers-please/blob/main/bin/common/models.py) what the LLM's response should look like, and validating it with a few lines of code.

Second, I moved **all deterministic behavior out of the LLM**. Rather than trying to get the model to get correctly follow the instructions 100% of the time, move as many pieces as possible outside of it. For instance, by examining failure cases by hand, I learnt which preprocessing operations might be needed often. A common case was removing the leading and trailing backticks, as shown above. Another common case was failing to produce a valid JSON, which led me to refine the prompt by being more explicit about the expected format. Similarly, when prioritizing articles it would ignore my instructions and bump articles even when the scoring system didn't warrant it. It was more efficient to let the LLM simply compute the score, and let me threshold it after the fact.

Third, the pipeline needs to be **robust to errors** caused by unexpected responses. In particular, LLM-powered steps are allowed to fail a few times before giving up. Furthermore, because API calls are precious, each step tries to salvage the parts of the batched-response that were valid, and only re-requests the invalid parts. But if the same input always produces the wrong answer, we will need to examine it by hand.

Which leads us to, fourth and last, a **modular design** was key. Each step is isolated from the others, so that debugging and fixing issues is straightforward. And thorough logging within each step ensured swift debugging.

## An agentic workflow that makes agentic workflows?

While developing `nf-papers-please`, some of the steps felt quite repetitive, yet required some modicum of intelligence to navigate. One example was going over all the failure cases of the LLM, and trying to understand what they had in common. It involved parsing large log files, navigating into individual working directories, and parsing the logs and error messages.

So I built a small agent to help me with that, in this case based on GitHub Copilot, to easily integrate into my VSCode workflow. To that end, I created an [AGENTS.md file](https://agents.md/) that gives the Agent context about the project as a whole, and a smaller file describing the specific tasks I wanted the agent to perform.

# Conclusions

While writing [`nf-papers-please`](https://github.com/hclimente/nf-papers-please), I often got frustrated. It was not only that LLM was not following my instructions. It was also that the community is still figuring out how to tackle these problems. On the painful side, the major players are pushing their own libraries and standards, changind their interfaces often while documentation is trying to catch up. But, on the bright side, it is exciting to see a new paradigm in computer science emerge. If you are willing to stay flexible and up-to-date, my bet is that mastering agentic workflows can really pay off in the long run. Otherwise, get as far as you can, and give it a few years until the community has converged into well built libraries.

# Further reading

- [12-Factor Agents - Principles for building reliable LLM applications](https://github.com/humanlayer/12-factor-agents)
