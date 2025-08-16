---
layout: post
title: Catalog of Algorithms
date: 2024-02-24 11:59:00-0400
description: Some of the best-known algorithms in computer science
tags:
  - computer_science
  - algorithms
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# Algorithms

## Divide and conquer problems

Divide and conquer algorithms work by breaking down a problem into _two or more_ smaller subproblems of the same type. These subproblems are tackled recursively, until the subproblem is simple enough to have a trivial solution. Then, the solutions are combined in a bottom-up fashion. For examples in [sorting](#sorting-problems), see [merge sort](#merge-sort) and [quick sort](#quick-sort).

## Intervals and scheduling problems

The input of interval problems is a list of lists, each of which contains a pair `[start_i, end_i]` representing an interval. Typical questions revolve around how much they overlap with each other, or inserting and merging a new element.

**Note:** There are many corner cases, like no intervals, intervals which end and start at the same time or intervals that englobe other intervals. Make sure to think it through.

**Note:** If the intervals are not sorted, the first step is _almost always_ **sorting them**, either by start or by end. This usually brings the time complexity to $$O(n \log n)$$. In some cases we need to perform two sorts, by start and end separately, before merging them. This produces the sequence of events that are happening.

## Sorting problems

Sorting consists on arranging the elements of an input array according to some criteria. There are multiple ways to sort an input, each offerintg different trade-offs:

- Memory usage: _in-place_ approaches sort the items in place, without using extra space.
- Stability: stable algorithms preserve the original relative order when faced with two equal keys.
- Internal vs external: internal sorts operate exclusively on RAM memory; external sorts do it outside (e.g., disk or tape).
- Recursive vs non-recursive
- Comparison-based: comparison-based algorithms work by comparing pairs of items. All the algorithms I cover here fall under this category, but not all (e.g., [counting sort](https://en.wikipedia.org/wiki/Counting_sort)).

I implement a couple of those below. Their complexities are as follows:

| Algorithm                        | Time complexity           | Space complexity |
| -------------------------------- | ------------------------- | ---------------- |
| [Selection](#selection-sort)     | $$O(n^2)$$                | $$O(1)$$         |
| [Bubble](#bubble-sort)           | $$O(n^2)$$                | $$O(1)$$         |
| [Merge](#merge-sort)             | $$O(n \log n)$$           | $$O(n)$$         |
| [Quicksort](#quick-sort)         | $$O(n \log n)$$ (average) | $$O(\log n)$$    |
| [Topological](#topological-sort) | $$O(\|V\| + \|E\|)$$      | $$O(\|V\|)$$     |

### Selection sort

```python
def selection_sort(x):

    for i in range(len(x)):
        curr_max, curr_max_idx = float("-inf"), None

        for j in range(len(x) - i):
            if x[j] > curr_max:
                curr_max = x[j]
                curr_max_idx = j

        x[~i], x[curr_max_idx] = x[curr_max_idx], x[~i]

    return x

bubble_sort([3,5,1,8,-1])
```

### Bubble sort

```python
def bubble_sort(x):
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
    return x

bubble_sort([3,5,1,8,-1])
```

### Merge sort

```python
def merge_sort(x):

    # base case
    if len(x) <= 1:
        return x

    # recursively sort the two halves
    mid = len(x) // 2
    sorted_left = merge_sort(x[:mid])
    sorted_right = merge_sort(x[mid:])

    # merge the two sorted halves
    i = j = 0
    merged = []

    while i < len(sorted_left) and j < len(sorted_right):
        if sorted_left[i] < sorted_right[j]:
            merged.append(sorted_left[i])
            i += 1
        else:
            merged.append(sorted_right[j])
            j += 1

    # since slicing forgives out of bounds starts
    # hence, this will work when i >= len(sorted_left)
    merged.extend(sorted_left[i:])
    merged.extend(sorted_right[j:])

    return merged


merge_sort([3,5,1,8,-1])
```

### Quick sort

```python
def quick_sort(x):

    if len(x) <= 1:
        return x

    pivot = x[-1] # preferrable to modifying the input with x.pop()
    lower = []
    higher = []

    # populate lower and higher in one loop,
    # instead of two list comprehensions
    for num in x[:-1]:
        if num <= pivot:
            lower.append(num)
        else:
            higher.append(num)

    return quick_sort(lower) + [pivot] + quick_sort(higher)

quick_sort([3,5,1,8,-1])
```

### Further reading

- [Sorting Out The Basics Behind Sorting Algorithms](https://medium.com/basecs/sorting-out-the-basics-behind-sorting-algorithms-b0a032873add)

## Linked lists

### Traversal

Traversing a linked list simply consists on passing through every element. We can do that starting from the head, following the pointer to the next node and so on.

For instance, this algorithm stores all the values into an array:

```python
class Node:

    def __init__(self, val):
        self.val = val
        self.next = None


def create_list():
    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")
    a.next = b
    b.next = c
    c.next = d
    return a

def fetch_values(head):

    curr = head
    values = []

    while curr:
        values.append(curr.val)
        curr = curr.next

a = create_list()
fetch_values(a)
```

```
['A', 'B', 'C', 'D']
```

Or recursively:

```python
def fetch_values(node):
    if not node: return values
    return [node.val] + fetch_values(node.next)


fetch_values(a)
```

```
['A', 'B', 'C', 'D']
```

### Search

```python
def find_value(node, target):

    if not node: return False
    elif node.val == target: return True

    return find_value(node.next, target)


find_value(a, "A") # True
find_value(b, "A") # False
```

### Keeping multiple pointers

Often multiple pointers are needed in order to perform certain operations on the list, like reversing it or deleting an element in the middle.

```python
def reverse_list(head):

    left, curr = None, head

    while curr:
        right = curr.next
        curr.next = left
        left, curr = curr, right

    return left


fetch_values(reverse_list(a))
```

```
['D', 'C', 'B', 'A']
```

### Merge lists

```python
a = create_list()

x = Node("X")
y = Node("Y")

x.next = y

def merge(head_1, head_2):

    tail = head_1
    curr_1, curr_2 = head_1.next, head_2
    counter = 0

    while curr_1 and curr_2:

        if counter & 1:
            tail.next = curr_1
            curr_1 = curr_1.next
        else:
            tail.next = curr_2
            curr_2 = curr_2.next

        tail = tail.next
        counter += 1

    if curr_1: tail.next = curr_1
    elif curr_2: tail.next = curr_2

    return head_1


fetch_values(merge(a, x))
```

```
['A', 'X', 'B', 'Y', 'C', 'D']
```

### Fast and slow pointers

Using two pointers that iterate the list at different speeds can help with multiple problems: finding the middle of a list, detecting cycles, or finding the element at a certain distance from the end. For instance, this is how you would use this technique to find the middle node:

```python
def find_middle(head):
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    return slow.val


a = create_list()
print(find_middle(a))
```

## Search problems

{% comment %}

### Linear search

TODO

### Binary search

TODO

## Tree problems

TODO

### Tree traversal

TODO
{% endcomment %}

#### In-order traversal

A very useful algorithm to know is how to iterate a BST in order, from the smallest to the largest value in the tree. It has a very compact recursive implementation:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    if root:
        return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)
    else:
        return []
```

However, a non-recursive implementation might be more easily adaptable to other problems:

```python
def inorder_traversal(root):

    output = []
    stack = []

    while root or stack:

        while root:
            stack.append(root)
            root = root.left

        root = stack.pop()
        output.append(root.val)
        root = root.right

    return output
```

For instance, to finding the k-smallest element:

```python
def find_k_smallest(root, k):

    stack = []

    while root or stack:

        while root:
            stack.append(root)
            root = root.left

        root = stack.pop()
        k -= 1

        if k == 0:
            return root.val

        root = root.right

    return None


# Construct the BST
#       3
#      / \
#     1   4
#      \
#       2
root = TreeNode(3)
root.left = TreeNode(1)
root.right = TreeNode(4)
root.left.right = TreeNode(2)

find_k_smallest(root, 2)
```

```
2
```

{% comment %}

### Search and delete

TODO

### Insert

TODO
{% endcomment %}

## Graph problems

### Traversals

The bread and butter of graph problems are traversal algorithms. Let's study them.

#### Depth first traversal

In a depth-first traversal (DFT), given a starting node, we recursively visit each of its neighbors before moving to the next one. In a 2D grid, it would involve picking a direction, and following it until we reach a bound. Then we would pick another direction, and do the same. Essentially, the exploration path looks like a snake.

The data structure underlying DFT is a **stack**:

1. When we visit a node, we push all of its neighbors. Hence, each frame in the stack is a node to visit.
2. We pop from the stack to visit the next node. Then we add its neighbors to the stack and continue.
3. Once we can't go deeper, pop will retrieve the last, unvisited branching point.
4. Once the stack is empty, our job is done.

Let's see an explicit implementation of the stack:

```python
graph = {
    "a": {"b", "c"},
    "b": {"d"},
    "c": {"e"},
    "d": {"f"},
    "e": set(),
    "f": set(),
}

def depth_first_print(graph: dict[str, set[str]], seed: str) -> None:
    stack = [seed]

    while stack:
        curr_node = stack.pop()
        print(curr_node)
        stack.extend(graph[curr_node])

depth_first_print(graph, "a")
```

```
a
b
d
f
c
e
```

Alternatively, we can use a recursive approach and an implicit stack:

```python
def depth_first_print(graph: dict[str, set[str]], seed: str) -> None:
    print(seed)
    for neighbor in graph[seed]:
        depth_first_print(graph, neighbor)

depth_first_print(graph, "a")
```

```
a
c
e
b
d
f
```

For a graph with nodes $$V$$ and edges $$E$$, the time complexity is $$O(\|V\|+\|E\|)$$ and the space complexity is $$O(\|V\|)$$.

**Note:** Watch out for _cycles_. Without explicing handling, we might get stuck in infinite traversals. We can keep track of which nodes we have visited using a set, and exit early as soon as we re-visit one.

**Note:** Some corner cases are the empty graph, graphs with one or two nodes, graphs with multiple components and graphs with cycles.

#### Breadth first traversal

In a breadth-first traversal (BFT), given a starting node, we first visit its neighbors, then their neighbors, and so on.

In a 2D grid, it doesn't favour any direction. Instead, it looks like a water ripple.

The data structure underlying BFT is a **queue**:

1. When we visit a node, we push all of its neighbors to the queue. As in DFT, each item is a node to visit.
2. We popleft to get the next node. We push allof its neighbors.
3. As before, once the queue is empty, our job is done.

Let's see an implementation:

```python
graph = {
    "a": {"b", "c"},
    "b": {"d"},
    "c": {"e"},
    "d": {"f"},
    "e": set(),
    "f": set(),
}

from collections import deque

def breadth_first_print(graph: dict[str, set[str]], seed: str) -> None:
    queue = deque([seed])

    while queue:
        curr_node = queue.popleft()
        print(curr_node)
        queue.extend(graph[curr_node])

breadth_first_print(graph, "a")
```

```
a
b
c
d
e
f
```

For a graph with nodes $$V$$ and edges $$E$$, the time complexity is $$O(\|V\|+\|E\|)$$ and the space complexity is $$O(\|V\|)$$.

### Topological sort

A topological sort (or _top sort_) is an algorithm whose input is a DAG, and whose output is an array such that every node appears after all the nodes that point at it. (Note that, in the presence of cycles, there is no valid topological sorting.) The algorithm looks like this:

1. Compute the indegree of every node, store it in a hash map.
1. Identify a node with no inbound edges in our hash map.
1. Add the node to the ordering.
1. Decrement the indegree of its neighbors.
1. Repeat from 2 until there are no nodes without inbound edges left.

Put together, the time complexity of top sort is $$O(\|V\| + \|E\|)$$, and the space complexity, $$O(\|V\|)$$.

{% comment %}

### Union find

TODO

### Djikstra

TODO

### Min spanning tree

TODO
{% endcomment %}

## Binary tree problems

### Tree traversals

As for graph related problems, problems involving trees often require traversals, either [depth](#depth-first-traversal) or [breadth](#breadth-first-traversal) first. The same principles and data structures apply. For a tree with $$n$$ nodes, the time complexity is $$O(n)$$, and the time complexity is $$O(n)$$. If the tree is balanced, depth first has a space complexity of $$O(\log n)$$.

### Further resources

- [Graph Algorithms for Technical Interviews - Full Course](https://www.youtube.com/watch?v=tWVWeAqZ0WU)

## Two pointers

The two pointer approach can be used in problems involving searching, comparing and modifying elements in a sequence. A naive approach would involve two loops, and hence take $$O(n^2)$$ time. Instead, in the two pointer approach we have two pointers storing indexes, and, by moving them in a coordinate way, we can reduce the complexity down to $$O(n)$$. Generally speaking, the two pointers can either move in the same direction, or in opposite directions.

**Note:** Some two pointer problems require the sequence to be sorted to move the pointers efficiently. For instance, to find the two elements that produce a sum, having a sorted array is key to know which pointer to increase or decrease.

**Note:** Sometimes we need to iterate an $$m \times n$$ table. While we can use two pointers for that, we can to with a single pointer $$i \in [0, m \times n)$$: `row = i // n`, `col = i % n`.

#### Sliding window problems

Sliding window problems are a type of same direction pointer problems. They are optimization problems involving **contiguous** sequences (substrings, subarrays, etc.), particularly involving cumulative properties. The general approach consists on starting with two pointers, `st` and `ed` at the beginning of the sequence. We can keep track of the cumulative property and update it as the window expands or contracts. We keep increasing `st` until we find a window that meets our constraint. Then, we try to reduce it by increasing `st`, until it doesn't meet it anymore. Then, we go back to increasing `ed`, and so on.

## Permutation problems

Permutation problems can be tackled by [recursion](#recursion).

## Backtracking problems

Backtracking is a family of algorithms characterized by:

- The candidate solutions are built incrementally.
- The solutions have **constraints**, so not all candidates are valid.

Since solutions are built incrementally, backtracting they can be visualized as a **depth-first search** on a tree. At each node, the algorithm checks if it will lead to a valid solution. If the answer is negative, it will _backtrack_ to the parent node, and continue the process.

**Note:** Because of the need to backtrack, a recursive implementation of the DFS is often more convenient, since undoing a step simply involves invoking `return`. A stack might require a more elaborate implementation.

### A recipe for backtracking problems

As we will see in a few examples, the solution to a backtracking problem looks like this:

```python
def solve(candidate):

    if is_solution(candidate):
        output(candidate)
        return

    for child in get_children(candidate):
        if is_valid(child):
            place(child)
            solve(child)
            remove(child)
```

### Examples

#### The eight queens puzzle

A famous application of backtracking is solving the [eight queens puzzle](https://en.wikipedia.org/wiki/Eight_queens_puzzle):

> The eight queens puzzle is the problem of placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other; thus, a solution requires that no two queens share the same row, column, or diagonal. There are 92 solutions.

I present here a solution, which mirrors the recipe presented above:

```python

board = []

def under_attack(row, col):
    for row_i, col_i in board:
        if row_i == row or col_i == col:
            return True

        # check the diagonals
        if abs(row_i - row) == abs(col_i - col):
            return True

    return False

def eight_queens(row=0, count=0):

    if row == 8:
        return count + 1

    for col in range(8):
        # check the constraints: the explored square
        # is not under attack
        if not under_attack(row, col):
            board.append((row, col))
            # explore a (so-far) valid path
            count = eight_queens(row + 1, count)
            # backtrack!
            board.pop()

    return count

total_solutions = eight_queens()
print(f"Total solutions: {total_solutions}")
```

```
Total solutions: 92
```

#### Solving a sudoku

```python

from pprint import pprint

board = [[0, 0, 0, 1, 0, 0, 0, 0, 5],
         [0, 0, 0, 0, 0, 4, 0, 1, 0],
         [1, 0, 3, 0, 0, 8, 4, 2, 7],
         [0, 0, 1, 7, 4, 6, 0, 9, 0],
         [0, 0, 6, 0, 3, 2, 1, 0, 8],
         [0, 3, 2, 5, 8, 0, 6, 0, 4],
         [0, 0, 7, 8, 0, 0, 0, 4, 0],
         [0, 0, 5, 0, 2, 7, 9, 8, 0],
         [0, 0, 0, 4, 6, 0, 0, 0, 0]]


def is_valid(board, row, col, num):

    block_row, block_col = (row // 3) * 3, (col // 3) * 3

    for i in range(9):
        if board[row][i] == num:
            return False
        elif board[i][col] == num:
            return False
        if board[block_row + i // 3][block_col + i % 3] == num:
            return False

    return True


def solve(board):

    for row in range(9):
        for col in range(9):

            if board[row][col]:
                continue

            for num in range(1, 10):
                if is_valid(board, row, col, num):
                    board[row][col] = num
                    if solve(board):
                        return True
                    board[row][col] = 0

            return False

    return True

if solve(board):
    pprint(board)
else:
    print("No solution exists.")
```

```
[[2, 7, 4, 1, 9, 3, 8, 6, 5],
 [6, 5, 8, 2, 7, 4, 3, 1, 9],
 [1, 9, 3, 6, 5, 8, 4, 2, 7],
 [5, 8, 1, 7, 4, 6, 2, 9, 3],
 [7, 4, 6, 9, 3, 2, 1, 5, 8],
 [9, 3, 2, 5, 8, 1, 6, 7, 4],
 [3, 2, 7, 8, 1, 9, 5, 4, 6],
 [4, 6, 5, 3, 2, 7, 9, 8, 1],
 [8, 1, 9, 4, 6, 5, 7, 3, 2]]
```

#### Permutations of a list

```python
def permute(nums):

    res = []
    size = len(nums)

    if not size: return [[]]

    for i in range(size):
        # exclude element i
        rest = nums[:i] + nums[i+1:]
        perms = [[nums[i]] + x for x in permute(rest)]
        res.extend(perms)

    return res
```

## Dynamic programming

The hallmark of a dynamic programming problem are **overlapping subproblems**.

The key to the problem is identifying the _trivially_ smallest input, the case for which the answer is trivially simple.

We have two strategies:

- Memoization
- Tabulation

Draw a strategy!!

### Recursion + memoization

#### Recursion

Recursion is a technique in to solve problems which in turn depend on solving smaller subproblems. It permeates many other methods, like [backtracking](#backtracking-problems), [merge sort](#merge-sort), [quick sort](#quick-sort), [binary search](#binary-search) or [tree traversal](#tree-traversal).

Recursive functions have two parts:

1. Definition of the **base case(s)**, the case(s) in which solving a problem is trivial, and a solution is provided, stopping the recursion.
1. Divide the problem into smaller subproblems, which are sent off to the recursive function.

The space complexity of recursion will be, at least, the length of the stack which accumulates all the function calls.

**Note:** CPython's recursion limit is 1,000. This can limit to the depth of the problems we can tackle.

{% comment %}

#### Memoization

TODO
{% endcomment %}

#### Recursion + memoization recipe

In DP, combining recursion and memoization is a powerful way to trade space complexity for time complexity. Specifically, since problems are overlapping, it is likely we are solving the same subproblems over and over, which can get expensive due to recursion. Caching them can greatly improve the speed of our algorithm.

Here is a recipe for solving these problems (from [here](https://www.youtube.com/watch?v=oBt53YbR9Kk)):

1. Visualize the problem as a tree
1. Implement the tree using recursion, in which the leaves are the base cases. This will produce the brute force solution.
1. Test it for a few simple cases.
1. Memoize it!
   1. Add a memo dictionary, which keeps getting passed in the recursive calls
   1. Add the base cases to the dictionary
   1. Store the return values into the memo
   1. Return the right value from memo

#### Computational complexity

The computational complexity will be impacted by two factors:

- `m`: the average length of the elements of the input. For instance, if the input is a list, `m = len(input)`; it it is an integer, it is `m = input`. This will impact the height of the tree.
- `n`: the length of the input. This will impact the branching factor. For instance, if the input is a list, `n = len(input)`.

**Brute force:** for every node, we have a `n` options. Usually, the time complexity of DP problems will be exponential, of $$O(n^m*k)$$, where $k$ is the complexity of a single recursive call. The memory complexity is the call stack, $$O(m)$$.

**Memoized:** memoization reduces the branching factor by storing previous results. In other words, it trades time complexity for space complexity; usually both become polynomial.

{% comment %}

### Tabulation

TODO
{% endcomment %}

#### Tabulation recipe

Taken from [here](https://www.youtube.com/watch?v=oBt53YbR9Kk):

1. Visualize the problem as a table. Specifically:
   1. Design the size of the table based on the inputs. Often the size of the table is one unit longer in each dimension than the respective inputs. That allows us to include the trivial case (usually in the first position), and nicely aligns our input with the last index.
   1. Pick the default value, usually based on what the output value should be.
   1. Infuse the trivial answer into the table, the case for which we immediately know the answer
1. Iterate through the table, filling the positions ahead based on the current position.
1. Retrieve the answer from the relevant position.

Some caveats:

1. Note that sometimes the trivial case might not have the solution we need to solve the algorithm. Watch out for such situations.

### Additional resources

These are some materials that helped me understand dynamic programming (the order matters!):

1. [A graphical introduction to dynamic programming](https://avikdas.com/2019/04/15/a-graphical-introduction-to-dynamic-programming.html)
1. [Dynamic Programming - Learn to Solve Algorithmic Problems & Coding Challenges](https://www.youtube.com/watch?v=oBt53YbR9Kk)
1. [Dynamic Programming is not Black Magic](https://qsantos.fr/2024/01/04/dynamic-programming-is-not-black-magic/)
1. [LeetCode: DP for Beginners](https://leetcode.com/discuss/study-guide/662866/DP-for-Beginners-Problems-or-Patterns-or-Sample-Solutions)

### Solved problems

```python
def how_sum(target: int, nums: list[int], memo: dict = {}) -> None | list[int]:
    if target == 0: return []
    if target < 0: return None
    if target in memo.keys(): return memo[target]
    for num in nums:
        solution = how_sum(target - num, nums, memo)
        if solution is not None:
            memo[target] = solution + [num]
            return memo[target]
    memo[target] = None
    return None

how_sum(300, [7, 14])
```

```python
def best_sum(target: int, nums: list[int], memo: dict = {}) -> None | list[int]:

    if target in memo: return memo[target]
    if target == 0: return []
    if target < 0: return None

    memo[target] = None
    length_best_solution = float("inf")

    for num in nums:
        solution = best_sum(target - num, nums, memo)

        if solution is not None and len(solution) < length_best_solution:
            memo[target] = solution + [num]
            length_best_solution = len(memo[target])

    return memo[target]

print(best_sum(7, [5, 3, 4, 7]))
print(best_sum(8, [1, 4, 5]))
print(best_sum(100, [1, 2, 5, 25]))
```

```python
def can_construct(target: str, dictionary: list, memo: dict = {}) -> bool:

    if target in memo: return memo[target]
    if not target: return True

    memo[target] = False

    for word in dictionary:
        if target.startswith(word):
            new_target = target.removeprefix(word)
            if can_construct(new_target, dictionary, memo):
                memo[target] = True
                break

    return memo[target]

print(can_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]))
print(can_construct("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"]))
print(can_construct("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef", ["e", "ee", "eee", "eeee", "eeeee", "eeeee"]))
```

```python
def count_construct(target: str, dictionary: list, memo: dict = {}) -> int:

    if target in memo: return memo[target]
    if not target: return 1

    memo[target] = 0

    for word in dictionary:
        if target.startswith(word):
            new_target = target.removeprefix(word)
            memo[target] += count_construct(new_target, dictionary, memo)

    return memo[target]

print(count_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]))
print(count_construct("purple", ["purp", "p", "ur", "le", "purpl"]))
print(count_construct("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"]))
print(count_construct("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef", ["e", "ee", "eee", "eeee", "eeeee", "eeeee"]))
```

```python
def all_construct(target: str, dictionary: list, memo: dict = {}) -> list[list[str]]:

    if target in memo: return memo[target]
    if not target: return [[]]

    memo[target] = []

    for word in dictionary:
        if target.startswith(word):
            new_target = target.removeprefix(word)
            constructs = all_construct(new_target, dictionary, memo)
            constructs = [[word] + c for c in constructs]
            memo[target].extend(constructs)

    return memo[target]

print(all_construct("abcdef", ["ab", "abc", "cd", "def", "abcd", "ef", "c"]))
print(all_construct("purple", ["purp", "p", "ur", "le", "purpl"]))
print(all_construct("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"]))
print(all_construct("eeeeeeeeeeeeeeeeeeeeef", ["e", "ee", "eee", "eeee", "eeeee", "eeeee"]))
```

````python
def fib_t(n: int) -> int:

    table = [0] * (n + 2)
    table[1] = 1

    for i in range(n):
        table[i + 1] += table[i]
        table[i + 2] += table[i]

    return table[n]

print(fib_t(6))
print(fib_t(50))

```python
def grid_traveler(m: int, n: int) -> int:

    grid = [[0] * (n + 1) for _ in range(m + 1)]
    grid[1][1] = 1

    for i in range(m + 1):
        for j in range(n + 1):
            if (i + 1) <= m:
                grid[i + 1][j] += grid[i][j]
            if (j + 1) <= n:
                grid[i][j + 1] += grid[i][j]

    return grid[m][n]

print(grid_traveler(1, 1))
print(grid_traveler(2, 3))
print(grid_traveler(3, 2))
print(grid_traveler(3, 3))
print(grid_traveler(18, 18))
````

```python
def can_sum_t(target: int, nums: list) -> bool:
    """
    Complexity:
        - Time: O(m*n)
        - Space: O(m)
    """

    grid = [False] * (target + 1)
    grid[0] = True

    for i in range(len(grid)):
        if not grid[i]:
            continue

        for num in nums:
            if (i + num) <= len(grid):
                grid[i + num] = True

    return grid[target]

print(can_sum_t(7, [2 ,3])) # True
print(can_sum_t(7, [5, 3, 4])) # True
print(can_sum_t(7, [2 ,4])) # False
print(can_sum_t(8, [2, 3, 5])) # True
print(can_sum_t(300, [7, 14])) # False
```

```python
def how_sum_t(target: int, nums: list[int]) -> None | list[int]:
    """
    Complexity:
        - Time: O(m*n^2)
        - Space: O(m*n)
    """
    grid = [None] * (target + 1)
    grid[0] = []

    for i in range(len(grid)):
        if grid[i] is None:
            continue

        for num in nums:
            if (i + num) < len(grid):
                grid[i + num] = grid[i].copy()
                grid[i + num].append(num)

    return grid[target]


print(how_sum_t(7, [2 ,3])) # [2, 2, 3]
print(how_sum_t(7, [5, 3, 4, 7])) # [3, 4]
print(how_sum_t(7, [2 ,4])) # None
print(how_sum_t(8, [2, 3, 5])) # [2, 2, 2, 2]
print(how_sum_t(300, [7, 14])) # None
```

```python
def best_sum_t(target: int, nums: list[int], memo: dict = {}) -> None | list[int]:
    """
    Complexity:
        - Time: O(m*n^2)
        - Space: O(m^2)
    """

    grid = [None] * (target + 1)
    grid[0] = []

    for i in range(len(grid)):
        if grid[i] is None:
            continue

        for num in nums:
            if (i + num) < len(grid):
                if grid[i + num] is None or len(grid[i + num]) > len(grid[i]):
                    grid[i + num] = grid[i].copy()
                    grid[i + num].append(num)

    return grid[target]

print(best_sum_t(7, [2 ,3])) # [2, 2, 3]
print(best_sum_t(7, [5, 3, 4, 7])) # [7]
print(best_sum_t(7, [2 ,4])) # None
print(best_sum_t(8, [2, 3, 5])) # [5, 3]
print(best_sum_t(300, [7, 14])) # None
```

```python
def can_construct_t(target: str, words: list[str]) -> bool:
    """
    Complexity:
        - Time: O(m^2*n)
        - Space: O(m)
    """

    grid = [False] * (len(target) + 1)
    grid[0] = True

    for i in range(len(grid)):

        if not grid[i]:
            continue

        prefix = target[:i]
        for word in words:
            if (i + len(word)) >= len(grid):
                continue

            if target.startswith(prefix + word):
                grid[i + len(word)] = True

    return grid[len(target)]

print(can_construct_t("abcdef", ["ab", "abc", "cd", "def", "abcd"])) # True
print(can_construct_t("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"])) # False
print(can_construct_t("enterapotentpot", ["a", "p", "ent", "enter", "ot", "o", "t"])) # True
print(can_construct_t("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef", ["e", "ee", "eee", "eeee", "eeeee", "eeeee"])) # False
```

```python
def count_construct_t(target: str, words: list[str]) -> int:
    """
    Complexity:
        - Time: O(m^2*n)
        - Space: O(m)
    """
    grid = [0] * (len(target) + 1)
    grid[0] = 1

    for i in range(len(grid)):
        if not grid[i]:
            continue

        for word in words:
            if (i + len(word)) >= len(grid):
                continue

            prefix = target[:i]

            if target.startswith(prefix + word):
                grid[i + len(word)] += grid[i]

    return grid[len(target)]

print(count_construct_t("abcdef", ["ab", "abc", "cd", "def", "abcd"])) # 1
print(count_construct_t("purple", ["purp", "p", "ur", "le", "purpl"])) # 2
print(count_construct_t("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"])) # 0
print(count_construct_t("enterapotentpot", ["a", "p", "ent", "enter", "ot", "o", "t"])) # 4
print(count_construct_t("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef", ["e", "ee", "eee", "eeee", "eeeee", "eeeee"])) # 0
```

```python
from copy import deepcopy

def all_construct_t(target: str, words: list[str]) -> list[list[str]]:
    """
    Complexity:
        - Time: O(n^m)
        - Memory: O(n^m)
    """

    grid = [[] for _ in range(len(target) + 1)]
    grid[0] = [[]]

    for i in range(len(grid)):

        if not grid[i]:
            continue

        for word in words:
            if (i + len(word)) > len(grid):
                continue

            prefix = target[:i]

            if target.startswith(prefix + word):
                new_constructs = deepcopy(grid[i])
                for x in new_constructs:
                    x.append(word)

                if grid[i + len(word)]:
                    grid[i + len(word)].extend(new_constructs)
                else:
                    grid[i + len(word)] = new_constructs


    return grid[len(target)]

print(all_construct_t("abcdef", ["ab", "abc", "cd", "def", "abcd", "ef", "c"])) # [['ab', 'cd', 'ef'], ['ab', 'c', 'def'], ['abc', 'def'], ['abcd', 'ef']]
print(all_construct_t("purple", ["purp", "p", "ur", "le", "purpl"])) # [['purp', 'le'], ['p', 'ur', 'p', 'le']]
print(all_construct_t("skateboard", ["bo", "rd", "ate", "t", "ska", "sk", "boar"])) # []
print(all_construct_t("enterapotentpot", ["a", "p", "ent", "enter", "ot", "o", "t"])) # # [['enter', 'a', 'p', 'ot', 'ent', 'p', 'ot'], ['enter', 'a', 'p', 'ot', 'ent', 'p', 'o', 't'], ['enter', 'a', 'p', 'o', 't', 'ent', 'p', 'ot'], ['enter', 'a', 'p', 'o', 't', 'ent', 'p', 'o', 't']]
```
