---
layout: post
title: Python objects
date: 2024-01-07 11:58:00-0000
description: Everything is an object!
tags: python coding objects
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

An object is a data structure with an internal state (a set of variables) and a behavior (a set of functions). Everything in Python is an object: builtins, functions, classes, instances. Thus, improving our understanding of objects is the first step to mastering Python.

> Note: not _everything_ in Python is an object.

# Properties of an object

Every Python object has three builtin properties (a reference, a class, and a refcount) as well as additional, class-specific properties.

## Reference

A **reference** is a pointer, a way to access the memory address that stores the object. It can be associated to a name, or an element in a collection:

```python
# create a new integer object, and
# copy its reference to the name "a"
a = 1

# create a new integer object, and
# append its reference to the list "x"
x = list()
x.append(1)
```

We can retrieve the memory address using `id()`, represented as an integer:

```python
id(a)
```

```
4342270592
```

> The assignment operator (`=`) just copies the reference to an object, not the object itself. Similarly, the deletion keyword (`del`) never deletes an object, just the reference to it.

We can check if two names point to the same memory location using `is`:

```python
x = [1, 2, 3]
y = [1, 2, 3]
x_copy = x

# same reference?
assert x is x_copy
assert id(x) == id(x_copy)
assert x is not y

# same value?
assert x == y
```

## Class

A **class** is the type of the object (e.g., a float, or a string). Each object contains a pointer to its class, [as we will see below](#the-two-dictionaries-underlying-an-object). We can retrieve an object's class using the `type` function:

```python
type(1)
```

```
<class 'int'>
```

```python
type("1")
```

```
<class 'str'>
```

```python
type(1.)
```

```
<class 'float'>
```

```python
class Dog:
    pass
print(type(Dog()))
```

```
<class '__main__.Dog'>
```

We can use `isinstance` to verify if an object is an instance of a given class:

```python
assert isinstance(1, int)
assert isinstance("1", str)
assert isinstance(1., float)
```

## Refcount

The **refcount** is a counter that keeps track of how many references point to an object. Its value gets increased by 1 when, for instance, an object gets assigned to a new name. It gets decreased by 1 when a name goes out of scope or is explicitly deleted (`del`). When the refcount reaches 0, its object's memory will be [reclaimed by the garbage collector]({% post_url 2024-02-11-python-basics %}#memory-management-in-python).

In principle, we can access the refcounts of a variable using `sys.getrefcount`:

```python
x = []
sys.getrefcount(x)
```

```
2
```

Note that despite `x` being the only reference to that empty list, the output of `getrefcount` is 2. This is because the function itself contains a new reference to the list, hence temporarily increasing its refcount by 1.

Let's see another example:

```python
sys.getrefcount(True)
```

```
4294967295
```

I expected that a newly created `bool(True)` object would have a `refcount` of 1. However, the actual number is much higher. This is because `True`, `False`, `None` and [a few others](https://docs.python.org/3/library/constants.html) are **singletons**, that is, only one such object can exist. By caching these common objects, Python can skip repeatedly instantiating them. Different Python implementations can have additional singletons for common objects. For instance, CPython pre-caches integers from -5 to 256:

```python
# two integers are created
# and each is assigned a name
x = 256
y = 256

# but they both point to the same object
assert x is y

# two integers are created
# and each is assigned a name
x = 257
y = 257

# but they are not singletons
assert x is not y
```

## Class' properties and methods

Objects have additional properties and methods that encode their state and behaviors. For instance, the `float` class has an additional property that stores the numerical value, as well as multiple methods that enable algebraic operations.

# Objects are first-class citizens

Objects are first-class citizens in Python, meaning they can be treated like any other value. In other words, they can:

- Be assigned a name:

```python
def pretty_print(x: str):
    print(x.title() + ".")

pp = pretty_print
pp("hey there")
```

```
Hey there.
```

- Be passed as arguments:

```python
from typing import Callable

def format(x: str, formatter: Callable[[str], None]):
    formatter(x)

format("hey there", pretty_print)
```

```
Hey there.
```

- Be returned by other functions:

```python
def formatter_factory():
    return pretty_print

formatter_factory()("hey there")
```

```
Hey there.
```

# Copying objects

As mentioned above, the assignment operator `=` does not copy objects, only references. If we need to copy an object, we need to use the `copy` module. There are two types of copies: shallow and deep.

Shallow copies, made with `copy.copy`, duplicate the object. However any reference it stores will just get copied as a reference, i.e., the referenced object will not be duplicated.

```python
from copy import copy

x = [1, 2, [3, 4]]
# copy the two first integers
# but only a reference to the
# 3rd element
x_copy = copy(x)

x[2].append(5)

assert x[2] is x_copy[2]
```

Deep copies, made with `copy.deepcopy`, recursively duplicates the object and all the referenced objects.

```python
from copy import deepcopy

x = [1, 2, [3, 4]]
# copies the two first integers
# as well as the list
x_copy = deepcopy(x)

x[2].append(5)

assert x[2] is not x_copy[2]
assert x[2] != x_copy[2]
```

# Defining our own classes

Python allows us to define our own classes using the `class` keyword. New objects are instantiated using the class name. Let's see an example:

```python
class Animal:

    phylum = "metazoan"

    def __init__(self, species, weight):
        self.species    = species
        self.weight     = weight
        self.__favorite = True

    def eat(self):
        self.weight += 1
        print("chompchomp")

    def is_favorite(self):
        return self.__favorite

    def set_favorite(self, flag):
        if isinstance(flag, bool):
            self.__favorite = flag
        else:
            msg = "flag should be a bool;" \
                f"{type(flag)} found."
            raise Exception(msg)
```

This is a defining an animal that can only eat while making cute noises. The animal also has some attributes: a phylum, a species, a weight and a boolean denoting if its my favorite or not. By default, they all are! But I have added two functions to read and change boolean. Let's zoom in on this simple example to understand some interesting properties of Python objects.

## Private and protected attributes

In other languages, a class' attributes can be set as public (accessible to everyone), protected (only accessible within the class and subclasses) or as private (only accessible within the class). This is helpful to define what we expose to other classes and what should not be modified or readable externally.

Python emulates protected and private attributes by prepending one or two underscores respectively. In our `Animal` example, `Animal.__favorite` is a private attribute:

```python
moby = Animal("whale", 100000)
moby.__favorite
```

```
AttributeError: 'Animal' object has no attribute
'__favorite'. Did you mean: 'is_favorite'?
```

We can read the private attribute using the public function `Animal.is_favorite()`, and modify it using the private function `Animal.set_favorite()`:

```python
moby.is_favorite()
```

```
True
```

```python
# sorry, moby :(
moby.set_favorite(False)

moby.is_favorite()
```

```
False
```

However, is just a convention: underscores signal intent, but access is not strictly restricted: you can _always_ modify attributes from the outside. However, we need to put some extra effort and by-pass the setter and getter:

```python
print(moby._Animal__favorite)
```

```
False
```

```python
moby._Animal__favorite = True
print(moby._Animal__favorite)
```

```
True
```

Note that we can define a new, public `__favorite` attribute on the instantiated object, which is different from the `_Animal__favorite` attribute defined at the class level. Hence, this becomes possible, which has been a source of headaches in the past:

```python
moby.__favorite = False
print(moby._Animal__favorite)
```

```
True
```

```python
print(moby.__favorite)
```

```
False
```

## The two dictionaries underlying an object

Underlying every object there are two dictionaries. They are accessible, respectively, using `{instance}.__dict__` and `{Class}.__dict__`. The first one is an instance-specific dictionary containing its writable attributes:

```python
moby = Animal("whale", 100000)

print(moby.__dict__)
```

```
{'species': 'whale', 'weight': 100000, '_Animal__favorite': True}
```

Note that private attributes like `__favorite` appear with an altered name of the form `_{class name}{attribute}`.

Similarly, each class has its own dictionary, containing the data and functions used by all instances (class' methods, attributes defined at the class level, etc.):

```python
Animal.__dict__
```

```
mappingproxy({'__module__': '__main__',
              '__firstlineno__': 1,
              'phylum': 'metazoan',
              '__init__': <function Animal.__init__ at 0x1019277e0>,
              'eat': <function Animal.eat at 0x1019274c0>,
              'is_favorite': <function Animal.is_favorite at 0x101927c40>,
              'set_favorite': <function Animal.set_favorite at 0x101927b00>,
              '__static_attributes__': ('__favorite', 'species', 'weight'),
              '__dict__': <attribute '__dict__' of 'Animal' objects>,
              '__weakref__': <attribute '__weakref__' of 'Animal' objects>,
              '__doc__': None})
```

For instance, this is where the `Animal.eat()` method lives. This dictionary is shared by all the instances, which is why every non-static method requires the instance to be passed as the first argument. Under the hood, when we call an instance's method, Python finds the method in the class dictionary and passes the instance as first argument. But we can also do it explicitly:

```python
Animal.__dict__["eat"]()
```

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Animal.eat() missing 1 required positional argument: 'self'
```

```python
Animal.__dict__["eat"](moby)
```

```
chompchomp
```

Both dictionaries are linked by `instance.__class__`, which is assigned to the class object:

```python
assert moby.__class__.__dict__ == Animal.__dict__
```

As we saw, an attribute might exist in either dictionary. To find an attribute at runtime, Python will first search `instance.__dict__`; if unsuccessful, it will search `Class.__dict__`.

## `__slots__` helps with memory optimization

The instance's dictionary keeps the class flexible, allowing to add new attributes at any time:

```python
moby.medium = "water"
print(moby.__dict__)
```

```
{'species': 'whale',
 'weight': 100000,
 '_Animal__favorite': False,
 'medium': 'water'}
```

`__slots__` allows us to fix the possible attributes a priori, allowing Python to reserve the exact amount of memory needed and to bypass the creation of the dictionary:

```python
class EfficientAnimal:

    __slots__ = ["species", "weight", "__favorite"]
    phylum = "metazoan"

    def __init__(self, species, weight):
        self.species    = species
        self.weight     = weight
        self.__favorite = True

wasabi = EfficientAnimal("dog", 10)
wasabi.__dict__
```

```
AttributeError: 'EfficientAnimal' object has no
attribute '__dict__'. Did you mean: '__dir__'?
```

This mainly optimizes memory, though it also blocks new attributes, helping to prevent bugs caused by typos in variable names:

```python
wasabi.wwweight = 8
```

```
AttributeError: 'EfficientAnimal' object has no
attribute 'wwweight'
```

# Serialization

**Serialization** is the process of converting a Python object into a file format that can be used to reconstruct the object later. This allows us to store objects, make them persistent across executions and distribute them. As the name implies, serialization consists on transforming the object, living in bits and pieces across the computer memory, into a linear sequence of bytes.

Simple objects can be easily serialized into a text-based format like JSON:

```python
import json

person1 = {
    "name": "John",
    "surname": "Doe",
    "age": 45
}

# serialize
with open("obj.json", mode="w") as J:
    json.dump(person1, J)

# deserialize
with open("obj.json", mode="r") as J:
    person2 = json.load(J)

assert person1 == person2
```

Let's see what happens if we try to serialize a more complex object:

```python
import json

class Person:
    def __init__(self, name, surname, age):
        self.name = name
        self.surname = surname
        self.age = age

    def __eq__(self, other):
        return (
            self.name == other.name and
            self.surname == other.surname and
            self.age == other.age
        )

person1 = Person("John", "Doe", 45)

with open("obj.json", mode="w") as J:
    json.dump(person1, J)
```

```
TypeError: Object of type Person is not JSON serializable
```

In such cases we need functions that assist Python in (de)serializing the object:

```python
def person_encoder(obj):
    if isinstance(obj, Person):
        obj_dict = {
            "__person__": True,
            "name": obj.name,
            "surname": obj.surname,
            "age": obj.age
            }
        return obj_dict

    msg = f'Cannot serialize object of {type(obj)}'

    raise TypeError(msg)


def person_decoder(dct):
    if dct.get("__person__", False):

        name    = dct["name"]
        surname = dct["surname"]
        age     = dct["age"]

        return Person(name, surname, age)

    return dct

# serialize
with open("obj.json", mode="w") as J:
    json.dump(person1, J, default=person_encoder)

# deserialize
with open("obj.json", mode="r") as J:
    person2 = json.load(J, object_hook=person_decoder)

assert person1 == person2
```

JSON is attractive because it is human-readable and interoperable. However (de)serializing sophisticated objects using JSON can be pretty involved due to the need to define an encoder and a decoder. Binary serialization, like the one provided by `pickle`, is attractive, since it handles complex objects out of the box:

```python
import pickle

person1 = Person("John", "Doe", 45)

# serialize or "pickle"
with open("obj.pkl", mode="wb") as P:
    pickle.dump(person1, P)

# deserialize or "unpickle"
with open("obj.pkl", mode="rb") as P:
    person2 = pickle.load(P)

assert person1 == person2
```

An important downside is that pickle objects are able to execute arbitrary code during unpickling. Hence, unpickling untrusted files should be regarded as a security risk.

# Further reading

- D. Beazley, [Advanced Python Mastery](https://github.com/dabeaz-course/python-mastery)
- https://www.interviewbit.com/python-interview-questions/#freshers
- More on classes: <https://docs.python.org/3/tutorial/classes.html>
- More on `__slots__`: <https://wiki.python.org/moin/UsingSlots>
