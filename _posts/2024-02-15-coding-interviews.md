---
layout: post
title: A strategy for coding interviews
date: 2024-02-24 11:59:00-0400
description: Success not guaranteed
tags:
  - computer_science
  - coding
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Although maybe on their way out, coding interviews are still common for computational roles. They aim to assess our problem-solving skills, to write code and to communicate our thought process. A common format involves a live coding session with an interviewer. The goal is to solve a problem in a limited time, usually 45-60 minutes. Here I present a template to tackle these situations.

# 1. Problem statement

The interviewer might come with a written down problem statement. They might share it with us ahead of our meeting or right at the start.

1. Make sure you understand the problem:
   1. Paraphrase the problem back to them
   1. If examples (input-output pairs) are provided, walk through one of them
   1. Otherwise, generate a few examples and infer the expected output
1. Ask clarifying questions:
   1. About the input:
      - What are its data types? Is it sorted? Do we know the range of the integers? (Can they be negative?) A batch of a stream? Et cetera.
      - Expected input size: if they know it, might give an idea of the complexity we should aim for. For inputs of size 1 to 100, $$O(n^2)$$ is acceptable; for larger inputs, we should do better.
   1. About the edge cases: empty input, invalid, etc.
   1. Ask about the specific runtime our solution will need. That will be very useful to screen out solutions and algorithms.
1. If possible, draw, or at least visualize the problem.

# 2. Brainstorming

While it can be tempting to start implementing a solution right away, it is worth spending some time drafting the problem. After all, our interviewer will have given it some thought already, and could be able to point us in the right direction.

1. Try to match this problem to the problems you have seen. Here's a cheat sheet for [data structures]({% post_url 2024-02-15-data-structures %}):
   - Hash maps: if we need fast lookups
   - Graphs: if we are working with associated entities
   - Stacks and queues: if the input has a nested quality
   - Heaps: if we need to perform scheduling/orderings based on a priority
   - Trees and tries: if we need efficient lookup and storage of strings
   - Linked lists: if we require fast insertions and deletions, especially when order matters
   - Union-finds: if we're investigating the if sets are connected or cycles exist in a graph
     And here's a list of common [algorithms]({% post_url 2024-02-15-algorithms %}):
   - Depth-first search
   - Binary Search
   - Sorting Algorithms
1. Don't be shy! Let the interviewer hear out your thought process. They will surely appreciate knowing whats on your mind, and be able to chip in. Specially, if they do say something, _listen_. After all, they know the solution already!
1. Once you seem to have converged to a specific approach, state the main parts of the algorithm and make sure they understand and agree.
   - We might want to start with a suboptimal solution, as long as we let them know that we know that! Once we have that working, we can identify the main bottlenecks and go back to the drawing board.

# 3. Implementation

During the implementation phase, it might help to go from the big picture to the small picture. Start by defining the global flow of the program, calling unimplemented functions with clear names. This will allow you to make sure your proposal make sense before getting entangled in the specifics.

In order to allow our interviewer follow our logic, it is important that they can follow along:

- Make sure they are ok with us using additional dependencies. They might prefer to keep the algorithm lean!
- Explain why you are making each decision.
- If you realize your solution might not work, let them know. You might need to go back to brainstorming.
- Stick to common language conventions. For instance, in Python, stick to PEP8.
- Keep your code clean: avoid duplicated code, use helper functions, keep function and variable names understandable.
- Time is limited, so you might want to cut corners, e.g.:
  - Comments
  - Function typing
  - Checking off-by-one errors when iterating arrays
    However, let your interviewer know!

Once you have a working solution, revisit it:

- Scan the code for mistakes. For instance, when working with arrays, index errors are common.
- Compute the complexity of your code. This might hint at what could be improved. It might also highlight tradeoffs.
- Identify redundant work
- Identify overlapping and repeated computations. The algorithm might be sped up by memoization.

# 4. Testing and debugging

Once our solution is ready, it might be a good idea to give it a go. Simply call your function on a few examples. Consider:

- "Normal" inputs
- Trivial inputs
- Edge inputs
- Invalid inputs

If some examples fail, we need to debug our code. Throw in a few print statements, predict what you expect to see, and go for it.

# 5. Follow-ups

After successfully presenting a solution, our interviewer might have some follow-up questions:

- About our solution:
  - Time and space complexity? Usually, we should consider the worst case complexity, but if the amortized case is significantly better you should point it out.
  - Specific questions about the choice of algorithm, data structure, loops, etc.
  - What are possible optimizations?
    - While abstracting specific aspects into functions is helpful, it might also be less efficient (e.g., if we have to iterate the input multiple times instead of one).
    - Identify repeated computations.
  - Consider non-technical constraints, such as development time, maintainability, or extensibility.
- Identify the best theoretical time complexity. This involves considering what is the minimum number of operations involved. For instance if we need to visit every element, probably $$O(n)$$ is optimal.

> Some algorithms have some implicit and potentially unexpected behaviors. Visit the [algorithms]({% post_url 2024-02-15-algorithms %}) post, and `Ctrl + F` "Note:" in order to find some of them.

# Further reading

- ["Blind 75" problem set](https://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU)
- [Code templates](https://leetcode.com/explore/interview/card/cheatsheets/720/resources/4723/)
- [Top techniques to approach and solve coding interview questions](https://www.techinterviewhandbook.org/coding-interview-techniques/)
- [What kind of problem do I have?](https://sebinsua.com/algorithmic-bathwater#what-kind-of-problem-do-i-have)
