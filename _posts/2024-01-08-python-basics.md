---
layout: post
title: The Basics of Python
date: 2024-01-08 11:59:00-0000
description: Revisiting Python's properties
tags: python coding
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Let's start by defining some of Python's essential qualities as a programming language.

# Python is dynamically typed

A language is **statically typed** when variables have types, i.e., the type of the variables are checked before execution (usually at compilation). In contrast, in **dynamically typed** languages variable names do not have types, runtime values (objects) do, i.e., the variable types are checked during execution. Python belongs to the second class.

# (C)Python is interpreted

No programming language is either interpreted or compiled. A language is just a set of instructions to tell a computer how to perform tasks. And this language can be implemented in different ways. For instance, while Python's reference implementation is [CPython](https://github.com/python/cpython), other implementations exist, like [IronPython](https://ironpython.net/) or [PyPy](https://www.pypy.org/). It is a fact, however, that most implementations of Python are **interpreted**. For instance, CPython executes a Python script in a two-step process:

1. "Compile" the source code into a Python-specific lower level code (`*.pyc`, stored in `__pycache__`), called _bytecode_.
1. Execute the bytecode by the Python Virtual Machine. This is an infinite evaluation loop that goes over all the lines, containing a switch over all possible bytecode instructions.

> Step 1's "compilation" is qualitatively different from the compilation of so-called compiled languages. For starters compiling a C code results in a standalone executable. Another difference is that CPython's aims to going from source code to execution as quickly as possible. Hence, it spends little time performing optimizations that would result in faster runtimes. In contrast, C compilers spend a significant amount of time optimizing the final binary, resulting in faster programs.

# Python types

The Python interpreter comes with some **predefined types**. They are:

- Numeric Types (`int`, `float`, `complex`)
- Boolean Type (`bool`)
- Iterator Types
- Sequence Types (`list`, `tuple`, `range`)
- Text Sequence Type (`str`)
- Binary Sequence Types (`bytes`, `bytearray`, `memoryview`)
- Set Types (`set`, `frozenset`)
- Mapping Types (`dict`)
- Context Manager Types
- Type Annotation Types (`Generic Alias`, `Union`)
- Other Built-in Types (modules, classes, `None` and others)

Python has two kinds of data types, **mutable** and **immutable**, which respectively can and cannot be modified after being created. Examples of mutable data types are lists, dictionaries and sets; examples of immutable data types are integers, floats and tuples. Let's see how mutability works through an example:

```python
# an int(1) object is created and
# both x and y point at it
x = y = 1

assert x is y

# we change the value of x. since
# integers are immutable, a new
# int(x + 1) is created to store
# that value, and x is assigned
# that new reference
x += 1

# x and y do not point to the same
# object anymore
assert x != y
assert x is not y
```

Let's compare this behaviour to that of a mutable object:

```python
# a list is created and both x
# and y point at it
x = y = [1]

assert x is y

# we change the value of x
# since lists are mutable,
# the original list gets altered
x.append(2)

# x and y still refer to the
# same object
assert x == y
assert x is y
```

Immutability is leveraged to define singletons, which I discussed when [examining the refcount]({% post_url 2024-01-07-python-objects %}#refcount).

Mutability also has implications on [memory allocation](#memory-management-in-python). Python knows at runtime how much memory an immutable data type requires. However, the memory requirements of mutable containers will change as we add and remove elements. Hence, to add new elements quickly if needed, Python allocates more memory than is strictly needed, as we saw in our close look [lists and tuples]({% post_url 2024-01-20-python-lists %}#the-inner-workings-of-lists-and-tuples).

# Scopes and namespaces

A **namespace** is a mapping from names to objects. In fact, underlying a namespace there is a dictionary: its keys are symbolic names (e.g., `x`) and its values are the object they reference (e.g., an integer with a value of 8). During the execution of a typical Python program, multiple namespaces are created, each with its own lifetime. There are four types of namespaces:

- **Builtin** namespaces: it is created when the interpreter starts up. It contain names such as `print`, `int` or `len`.
- **Global** namespaces: _The_ global namespace contains every name created at the main level of the program. This dictionary can be examined using `globals()`. But, _other_ global namespaces are possible: each imported module will create its own.
- **Local** namespaces: one is created every time a function is called, and is "forgotten" when it terminates. This dictionary can be examined using `locals()`.
- **Enclosed** namespaces: when a function calls another function, the child has access to its parent's namespace.

Namespaces are related to scopes, which are the parts of the code in which a specific set of namespaces can be accessed. When a Python needs to lookup a name, if resolves it by examining the namespaces using the LEGB rule: it starts at the Local namespace; if unsuccessful, it moves to the Enclosing namespace; then the Global, and lastly the Builtin. By default, assignments and deletions happen on the local namespace. However, this behaviour can be altered using the `nonlocal` and `global` statements.

Let's tie it all together with an example:

```python
def f_enclosed():
	foo = "enclosed"

	# use foo defined within f_enclosed
	print("\tInside f_enclosed():")
	print(f"\tfoo = {foo}")

	def f_local():
		foo = "local"

		# use foo defined within f_local
		print("\t\tInside f_local():")
		print(f"\t\tfoo = {foo}")

	f_local()

	# f_local's foo is gone
	# we are back to f_enclosed's
	print("\tAfter f_local():")
	print(f"\tfoo = {foo}")

	def f_non_local():

		# modifies the foo from the
		# enclosing namespace
		# i.e., f_enclosed's
		nonlocal foo
		print("\t\tInside f_non_local():")
		print(f"\t\toriginally, foo = {foo}")

		foo = "non_local"
		print(f"\t\tbut then, foo = {foo}")

	f_non_local()

	# f_enclosed's foo is changed even
	# outside of f_non_local
	print("\tAfter f_non_local():")
	print(f"\tfoo = {foo}")

	def f_global():

		# modifies the foo from the
		# global namespace
		global foo
		print("\t\tInside f_global():")
		print(f"\t\toriginally, foo = {foo}")

		foo = "global"
		print(f"\t\tbut then, foo = {foo}")

	f_global()

	# f_enclosed's remains unchanged
	print("\tAfter f_global():")
	print(f"\tfoo = {foo}")

foo = "original"
print("At the beginning:")
print(f"foo = {foo}")
f_enclosed()
print("Finally:")
print(f"foo = {foo}")
```

```
At the beginning:
foo = original
	Inside f_enclosed():
	foo = enclosed
		Inside f_local():
		foo = local
	After f_local():
	foo = enclosed
		Inside f_non_local():
		originally, foo = enclosed
		but then, foo = non_local
	After f_non_local():
	foo = non_local
		Inside f_global():
		originally, foo = original
		but then, foo = global
	After f_global():
	foo = non_local
Finally:
foo = global
```

# Memory management in Python

In CPython, all the objects live in a _private_ [heap]({% post_url 2024-02-10-hardware %}#memory-allocation). Memory management is handled exclusively by the Python memory manager. In other words, and in contrast to languages like C, the user has no way directly to manipulate items in memory. The Python heap is further subdivided into _arenas_ to reduce data fragmentation.

When an object is created, the memory manager allocates some memory for it in the heap, and its reference is stored in the relevant namespace.

Conversely, the garbage collector is an algorithm that deallocates objects when they are no longer needed. The main mechanism uses the [reference count]({% post_url 2024-01-07-python-objects %}#refcount) of the object: when it falls to 0, its memory is deallocated. However, the garbage collector also watches for objects that still have a non-zero refcount, but have become inaccessible, for instance:

```python
# create a list
# refcount = 1
x = []

# add a reference to itself
# refcount = 2
x.append(x)

# delete the original reference
# refcount = 1
del x

# any reference to the list is lost
# the garbage collector will remove it
```

# The global interpreter lock

> Since Python 3.13, the GIL can be disabled

The global interpreter lock (GIL) is a mechanism to make CPython's thread safe, by only allows one thread to execute Python bytecode at a time. This vastly simplifies CPython's implementation and writing extensions for it, since thread safety is not a concern. It also leads to faster single-thread applications. However, CPU-bound tasks cannot be sped up by multithreading, since nonetheless the threads will run sequentially, never in parallel. However, it can be used to speed up I/O-bound operations.

When parallel processing is needed, Python can still do that via:

1. Multiprocessing, i.e., launching multiple Python processes, each with their own interpreter, memory, and GIL.
1. Developing a C extension, which gives us lower-level access to threading.

# The Python import system

A Python module is simply a file containing Python functions, classes, constants and runnable code. When we want to use them, we need to _import_ the module using the `import` statement. For instance:

```python
import numpy as np
```

It imports [this file](https://github.com/numpy/numpy/blob/main/numpy/__init__.py) from your installed NumPy package as a module object and assigns its reference name `np`.

There are multiple things that Python recognizes as modules:

1. Built-in modules: written in C, and part of the Python executable.
1. Frozen modules: written in Python, and part of the Python executable.
1. C extensions: written in C, but loaded dynamically into the Python executable.
1. Python source code and bytecode files
1. Directories

{% comment %}
TO EXPAND

# TODO

- Scope resolution
  {% endcomment %}

# Further reading

- [StackOverflow: If Python is interpreted, what are .pyc files?](https://stackoverflow.com/questions/2998215/if-python-is-interpreted-what-are-pyc-files)
- [Python behind the scenes #11: how the Python import system works](https://tenthousandmeters.com/blog/python-behind-the-scenes-11-how-the-python-import-system-works/)
- [Python behind the scenes #13: the GIL and its effects on Python multithreading](https://tenthousandmeters.com/blog/python-behind-the-scenes-13-the-gil-and-its-effects-on-python-multithreading/)
