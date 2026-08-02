---
layout: distill
title: Cryptography 101: Encryption
date: 2025-11-17 11:59:00 +0000
description: Ensuring confidentiality
tags:
  - algorithms
  - computer_science
  - cryptography
giscus_comments: true
related_posts: false
---

| __Key algorithms__ | AES, elliptic curve cryptography |
| __Key protocols__  | TLS/HTTPS, WPA3, IKE             |

The main tool to ensure that our communications remain private is __encryption__. Encryption consists on reversibly transforming a message into an (apparently) random message using an encyption key. The decryption key, if you have it, allows you to recover the original information.

We will first focus on __symmetric__ encryption, i.e., the same key is used for both actions. Then, we'll move onto __asymmetic__ encription, in which the keys are different.

# Symmetric Encription: Advanced Encryption Standard

The [Caesar cipher](https://en.wikipedia.org/wiki/Caesar_cipher) is the simplest example. Our key is a number, which indicates how many letters we shift the alphabet by:

| Original letter | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|-----------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Shift by 3      | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | A | B | C |

To encrypt the message "HELLO" with _key 3_, we shift each letter by 3 positions, resulting in "KHOOR". To decrypt, we simply shift back by 3.

Given a long enough message, breaking the Caesar cipher is easy. Since we know that "E", "T" and "A" are the most common letters in English texts, we expect the most common letters in the message to map to that. Or we can simply brute-force all 26 possibilities and see which one produces an intelligible text.

Luckily the field has evolved quite a bit in the last two millenia. The standard encryption algorithm nowadays is the __Advanced Encryption Standard__ (AES). Same as Caesar cipher, AES is a symmetric encryption algorithm. However the key is not a number between 0 and 25, but an enormously large integer, one that requires 128, 192, or 256 bits to represent. They respectively define the three flavors of AES: AES-128, AES-192, and AES-256. Longer keys provide more secure encryption, but make encryption and decryption more computationally intensive.

{% details What does it mean for a key to be secure? %}

In short, a key is secure when it cannot be guessed easily. In other words, it's secure when it's _long_, i.e., it can only be guessed via _brute force_. Since AES-128 has $$2^128$$ possible 128-bit keys, we can see how testing them all is unfeasible. The _algorithm_ is even stronger when testing each key is _expensive_.

{% enddetails %}

AES is pretty convoluted algorithm, and I find it a bit uninteresting. Kind of like an algorithm to shuffle cards reproducibly; interesting and lucrative applications, boring to watch. Luckily for me there are many good, detailed explanations around the web I can point you to (e.g., [this one](https://www.geeksforgeeks.org/computer-networks/advanced-encryption-standard-aes/)).

The TL;DR is this: AES starts by decomposing the message into chunks of 16 bytes, which are arranged into a 4-by-4 grid, each cell containing 1 byte. Each chunk is then processed independently in _rounds_. One round consists on a pretty complex transformation of the grid, involving dictionary replacements of the cells' contents, shifting rows and columns and, finally, a combination with a key. The key is round-specific, and is derived from the encryption key. The number of rounds depends on the length of the key (10 in AES-128, 12 in AES-192 and 14 in AES-256). Decrypting the data consists on performing the steps in reverse order.

> Encryption alone doesn't guarantee that our data hasn't been tampered with. An attacker can still flip bits and alter a message, even if they are not quite sure what they are doing. We explore how to guarantee __integrity__ in another post.

# Sharing keys: public key cryptography

AES is _everywhere_, and it is used gazillions of times every day to encrypt all hard drives (e.g., on MacOS).

But what about securing _communications_? How can Alice and Bob agree on an AES key in the presence of Eve, who will listen to every word they cross? In the old times, Alice and Bob would meet in a park and exchange keys in closed envelopes, making sure Eve can't get a peek. But in 1976 two researchers, Diffie and Hellman, introduced an algorithm that allowed them to agree on a key in public. This unlocked __public key cryptography__ and, ultimately, secure communications over the internet, like browsing the internet (implemented in TLS/HTTPS), securing our WiFi (WPA3), or using a VPN (IKE).

## Trapdoor functions

At the core of public key cryptography lies a [trapdoor function](https://en.wikipedia.org/wiki/Trapdoor_function), a mathematical function that's easy to do, but very hard to undo. Alice and Bob each apply have their own trapdoor function and, by only sharing their respective outputs, can reach the same mathematical result. Meanwhile, Eve will fall right through the trapdoor, taking her eons to figure out what the functions were.

A classic example of a trapdoor function, and the one used by Diffie-Hellman, is __modular exponentiation__.

$$
g^{k} \bmod p
$$

If tell I you the second hand of my clock ($$\text{Current time} \bmod 60$$) is pointing at 32 ($$g = 32$$) seconds, and that I raise that number to the 4th power ($$k = 4$$), you will have no issues computing that it will end up pointing at

$$
32^{4} \bmod 60 = 16.
$$

But if only I tell you I started at 32 and ended at 16, and ask you what $$k$$ was, the problem becomes much harder. This is known as the __discrete logarithm problem__, and you'll only be able to solve it by enumerating all possibilities:

$$
32^{1} \bmod 60 = 32
$$

$$
32^{2} \bmod 60 = 4
$$

$$
32^{3} \bmod 60 = 8
$$

$$
32^{4} \bmod 60 = 16
$$

Now, it turns out 60 is not a great choice for $$p$$. The space of outcomes encompasses, at most, the 60 ticks of the clock. That means that at most we need to enumerate 60 possibilities to identify the number $$k$$ is a multiple of, reducing our search space by a factor of 60. The larger $$p$$ is, the harder this problem becomes: we want $$\boldsymbol{p}$$ __to be astronomically large__; Diffie-Hellman makes it at least 2048 bits long.

But, given that we are doing modulo 60, 32 is not a good choice for $$g$$ either. Out of the 60 outcomes $$\operatorname{mod} 60$$ offers, the powers of 32 modulo 60 occupy only 4: the solution to $$k=1$$ is the same as to $$k=5$$: $$32^{5} \bmod 60 = 32$$. Hence, we quickly shrink our space of possibilities by a factor of 15: only multiples of 4 could produce a remainder of 16. We want the opposite: all options between 0 and $$p$$ should be possible (i.e., we want $$g$$ to be a __primitive root modulo__ $$\boldsymbol{p}$$.) While there are better choices than 32, [there are no primitive roots modulo 60](https://en.wikipedia.org/wiki/Primitive_root_modulo_n#:~:text=A%20primitive%20root%20exists%20if%20and,odd%20prime%20and%20k%20%3E%200.), and hence we should ditch it altogether. A better choice would be $$g=5$$ and $$p = 6$$, since 5 is a primitive root modulo 6. An even better choice would be to pick a massive, prime $$p$$, and a small $$g$$ that is a primitive root modulo $$p$$.

## The OG: the Diffie-Hellman algorithm

Now that we understand its trapdoor function, let's go back to Diffie-Hellman. Alice and Bob want to communicate privately with each other. To that end, they will encrypt each of their messages using AES-256. However, they haven't agreed on an encryption key yet, and Eve is there listening to everything they say to each other. (Not too unlike anytime we access a website on public WiFi.)

This is how Diffie and Hellman solved this issue:

1. Alice and Bob will first establish a communication using a pre-agreed protocol. The protocol determines the two integers $$g$$ and $$p$$ that will provide a good trapdoor $$f$$:

    $$
    f(k) = g^{k} \bmod p
    $$

1. Both Alice and Bob generate a secret large integer ($$k_A$$ and $$k_B$$, respectively) that they never share with each other (and hence with Eve). They will use it to define their respective trapdoor functions:

    $$
    z_i = f(k_i)
    $$

1. Alice and Bob share that result publicly ($$z_A$$ and $$z_B$$, respectively).
1. Alice and Bob leverage each other's intermediate result and their own private key to compute the same number:

    $$
    \begin{align*}
    z_A^{k_B} \bmod p &= (g^{k_A} \bmod p)^{k_B} \bmod p \\\\
    &= g^{k_A \cdot k_B} \bmod p \\\\
    &= (g^{k_B} \bmod p)^{k_A} \bmod p \\\\
    &= z_B^{k_A} \bmod p
    \end{align*}
    $$

    Crucially, Eve is clueless about which this number is.

1. Alice and Bob will pass that number to the same key derivation function to produce the AES key they'll use in their communications.

## Elliptic curve cryptography

A crucial drawback of Diffie-Hellman is the computational burden of integer arithmetic on astronomically large $$p\text{s}$$. This gets really expensive at scale! Consider the burden on servers that need to run all these computations on millions of concurrent connections. Luckily modern cryptography has mostly moved past Diffie-Hellman's modular exponentiation and into __elliptic curves__, which offer comparable security with much smaller key sizes.

Elliptic curves are the set of points satisfying an equation of the form

$$
y^2 = x^3 + ax + b,
$$

where $$a$$ and $$b$$ are parameters.

### The trapdoor

Elliptic curves are defined over a finite field of whole numbers. However, let's set that aside for now, and develop our intuitions on the continuous case.

Elliptic curves cryptography relies on __elliptic curve point addition and multiplication__. __Addition__ is the operation that allows us to combine elements and obtain a third one. Elliptic curve addition of two points consists on taking the line connecting them, intersecting it with the curve itself and taking the reflection of that point across the x-axis.

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_addition.gif" class="img-fluid" %}
<div class="caption" align="center">
    <b>Elliptic curve point addition.</b>
</div>

While this might seem like a weird way to define addition, note that it guarantees an additive inverse exists (e.g., subtracting $$P$$ from $$P+Q$$ produces $$Q$$):

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_subtraction.gif" class="img-fluid" %}
<div class="caption" align="center">
    <b>Elliptic curve point subtraction.</b>
</div>

__Multiplication__ is the operation that allows us to repeatedly perform addition. Elliptic curve multiplication consists on adding a point over and over. The main problem is defining $$2P = P + P$$ in the absence of another point to compute a line. To do that, we will define the first line taking the tangent of the curve at $$P$$. Then, we simply keep adding $$P$$ to the resulting number to compute higher multiples.

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_multiplication.gif" class="img-fluid" %}
<div class="caption" align="center">
    <b>Elliptic curve point multiplication.</b>
</div>

__Our trapdoor is elliptic curve point multiplication.__ Given a _base point_ $$P$$ and a factor $$k$$, it is easy to compute $$kP$$. However, given $$kP$$ and $$P$$, it is really hard to guess the value of $$k$$. $$P$$ is our public key, and $$k$$ is our secret key.

### Finite fields

Working with real numbers is hard for computers, since precision is lost. That's why cryptography prefers to work with integers. Remember I said elliptic curves are defined over a finite field? Let's revisit that.

We can define a curve as the set of solutions to

$$
y^2 \equiv x^3 + ax + b \pmod{p}
$$

where $$p$$ is a prime, defined over $$\mathbb{Z} / p \mathbb{Z}$$. It's hard to see a _curve_ here, although we should by analogy to the continuous case:

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_finite_field.png" class="img-fluid" %}

<div class="caption" align="center">
    <b>Elliptic curve on a finite field: \(y^2 \equiv x^3 - x + 1 \pmod {47}\).</b>
</div>

{% details What am I looking at? %}

First, the curve is only defined for integers. This means that we won't see a smooth continuous curve. It also means that for some values of $$x$$, $$y$$ is not an integer, and hence it's not defined. For instance, $$x = 20$$ would produce $$y_1 = 4.66$$ and $$y_2 = 42.34$$, which are outside of the codomain.

Second, we use a _congruence_ relationship ($$\equiv$$) rather than _equality_ ($$=$$). This is because we're working in a finite field: all values must stay within $$\{0, 1, 2, \ldots, p-1\}$$. Congruence means $$y^2$$ and $$x^3 + ax + b$$ don't need to be exactly equal, they just need to have the same remainder when divided by $$p$$. For example, $$17 \equiv 2 \pmod{5}$$ because both leave remainder 2 when divided by 5.

{% enddetails %}

We can also define addition $$P+Q$$ on this curve, although with two peculiarities. First, the line connecting $$P$$ and $$Q$$ is not a continuous line either, but a discrete set of points. First, we compute its slope:

$$m = \frac {y_Q - y_P} {x_Q - x_P} \pmod{p} $$

Then, we find the intersection with the curve:

$$x_{P+Q} = (m**2 - x_P - x2) \pmod{p} $$

$$y_{P+Q} = (m * (x_P - x_{P+Q}) - x_P) \pmod{p} $$

Second, the modulo warps the space, so when line reaches the top it continues from the bottom, and similarly from left to right. Putting it all together, it looks like this:

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_mod_addition.gif" class="img-fluid" %}
<div class="caption" align="center">
    <b>Elliptic curve point addition on a finite field.</b> The dashed blue line represents the continuous version of the (warped) line and the blue square markers represent the actual points on the line.
</div>

And, once we have addition, multiplication is just about doing that operation over and over.

### Curve25519

A widely used curve is so-called Curve25519:

$$
y^2 = x^3 + 486662 x^2 + x \pmod{2^{255} − 19}
$$

The public key is just the base point $$(9, 39\,420\,351)$$. The private key is a 256 intefer we multiply it by. Significantly shorter than the 2048-bit one that DH required!

# Asymmetric Encryption

TODO


# TL;DR: Everyday Encryption

Here are my (subjectively ranked) recommendations.

## Encrypt your data!

The first way to ensure confidentiality of my personal data private is to encrypt it, in case I lose my devices. If your data is unencrypted, basically anyone can take out the hard drive from your laptop and read its contents. In the Apple ecosystem, this involves:

- MacOS: enable FileVault (`System Settings > Privacy & Security > FileVault`) to encrypt you data using [a variant of AES-256](https://support.apple.com/en-gb/guide/security/sec4c6dc1b6e/web).
- iOS: by default, data is encrypted using AES.
- iCloud: Advanced Data Protection ensures that our data is encrypted _before_ being uploaded to iCloud with a key only you have. This ensures that even if someone gets our iCloud password they can't read it; in theory not even Apple can.

> Unfortunately, in the UK, His Majesty's Government needs full access to our data, and hence we cannot use this protection.

## Encrypt your communications

VPN

- Signal protocol: keeping a conversation secure.
