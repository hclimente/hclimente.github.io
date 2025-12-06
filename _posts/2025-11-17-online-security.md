---
layout: distill
title: Digital security
date: 2025-11-17 11:59:00 +0000
description: Cocktail parties make me anxious
tags:
  - algorithms
  - computer_science
  - cryptography
giscus_comments: true
related_posts: false
---

Most activities in our digital life can be broken down four three cryptographic goals:

- __Encryption__: ensure that our data and communications are private
- __Authentication__: ensure that we are who we say we are
- __Integrity__: ensure that our data and communications are not tampered with
- __Non-repudiation__: ensure that we cannot deny having authored a message

Let's see how each of them is relevant in our day-to-day online activities.

| Threat                 | Encryption | Authentication | Integrity | Non-repudiation |
|------------------------|------------|----------------|-----------|-----------------|
| Private communications | ✓          |                |           |                 |
| Phishing attacks       | ✓          | ✓              | ✓         |                 |
| Theft of devices       | ✓          | ✓              |           |                 |
| Malicious software     |            | ✓              | ✓         |                 |
| Data tampering         |            |                | ✓         |                 |
| Online transactions    | ✓          | ✓              |           | ✓               |
| SSH access             | ✓          | ✓              | ✓         |                 |
| `git commit`           |            |                | ✓         | ✓               |
| `git push`             | ✓          | ✓              | ✓         | ✓               |

In this post, I go over the main algorithms behind each goal, and how I use them to stay safe online. If you don't care about the theory, simply skip to the TL;DR of each section.

# Algorithm

- Keep passwords: SHA-256 is too fast. Hence, a hacker could eventually crack your password if your SHA gets leaked. Instead, we use Argon2.
- Signal protocol: keeping a conversation secure.

Protocols: which algorithm to use.

- TLS: negotiate a connection between browser and a server, using ECDSA to verify who you are, AES to encrypt the data, and SHA to verify that no packaets were lost.
- VPNs: same, but setting up a secure tunner between two networks.

# Encryption

Encrypting data (hard drive, network traffic): AES

## The cipher: Advanced Encryption Standard

AES is a 128-bit symmetric block cypher. The key has a fixed size (128, 192, 256 bits).

AES arranges the message into a grid.

1 round: a pretty complex transformation of the grid.

128 key: 10 rounds
192: 12 rounds
256: 14 rounds

## Sharing keys

If you and a friend were to exchange a secret key in public over the internet, anyone listening in could get it.

{% details Diffie-Hellman %}

The Diffie-Hellman exchange is a way to exchange secret keys in public. Instead of sharing the key publicly, DH shares the variables that are used to generate the secret key, two large integers $$x$$ and $$y$$. You each would generate a third large number (intermediate key), and apply modular exponentiation operation to it:

$$
z_1 = x^{k_1} \bmod y
$$

$$
z_2 = x^{k_2} \bmod y
$$

Then, you'd share that publicly. Notably, modular exponentiation is easy to do, but hard to undo. I.e., if $$x$$ and $$y$$ meet the right conditions, it's hard to recover $$k_1$$ just from $$z_1$$. In fact, the only way is to try all possible intermediate keys untl we stumble upon the right one.

However, we can leverage each other's intermediate result and our own private key to achieve the same number:

$$
\begin{align*}
  z_1^{k_2} \bmod y &= (x^{k_1} \bmod y)^{k_2} \bmod y \\\\
  &= x^{k_1 \cdot k_2} \bmod y \\\\
  &= (x^{k_2} \bmod y)^{k_1} \bmod y \\\\
  &= z_2^{k_1} \bmod y
\end{align*}
$$

This new number, is our __shared secret__.

For this to work:

- $$y$$ needs to be a large prime number
- $$x$$ needs to be a _primitive root modulo n_. In other words, $$x^\text{n} mod y$$ should produce all positive numbers between 0 and $$y$$.

The resulting key is often leveraged by AES, which allows to quickly encrypt and decrypt messages.

{% enddetails %}

$$
y^2 = x^3 + ax + b
$$

where $$a$$ and $$b$$ are parameters.

We have a generator $$g$$ which is a point on that curve. Then, we can define additions on the curve. $$2g = g + g$$, which is the result of taking the tangent of the curve at $$g$$, and taking its mirror image. $$3g$$ is the mirror image of the intercept between the curve and the line connecting $$g$$ and $$2g$$. $$4g = 3g + g$$ is the mirror image of the intersection of the curve with the line connecting 3g and g. And so on.

Adding points on an elliptic curve is a way to get points on the curve (apparently) at random. Given a point on the curve, that we know is $$xg$$, x is our secret key.

This is a replacement for DH. We add a modulo to it.

This is more complicated, but much more efficient mathematically: we can use much shorter keys, and hence to less operations. This is important server-side. The public key is a x, y point, although we can just use x.

## TL;DR: Encryption

# Authentication

Proving identity (prove who you say you are, e.g., to authenticate of exchange keys, credit card payments): ECDSA, Ed25519

The core of the problem is authentication. How can Gmail be sure that the person logging into my email account is really me? Usually, this is done by requesting information that only I should have. Usually, it's one or several of the following:

- Something I know, like a password, a PIN, or the answer to a secret question
- Something I have, like a phone, a hardware token, or a smart card
- Something I am, like a fingerprint, a facial scan, or a voiceprint

A cornerstone of good security is multi-factor authentication (MFA), which requires two or more of these factors to authenticate. That's why many important services these days require you to provide, e.g., both a password and a code sent to your phone.

Of course, the more factors you require, the safer you are. But security comes at the cost of convenience. Maybe you don't want to scan your face and receive an email code to shitpost on Reddit. Maybe you don't have good signal, and would rather keep text messages for the imporant stuff.

## FIDO2 and passkeys

Passkeys are the implementation of the FIDO2 authentication standard. During setup, your device (browser, phone, security key) creates two keys: a public key (`pk`) and a private key (`sk`).

## RSA: preventing Man-in-the-middle

Imagine there is someone in the middle, Sean. They perform the DH with both Alice and Bob, while Alice and Bob think they are talking to each other. This ends up in the situation in which both Alice and Bob share a secret with Sean, and none with each other. This is bad. Alice and Bob has no way to know they are not talking to each other.

DH is designed for two parties to exchange a secret in public, but not when there is someone intercepting the secrets and altering the messages.

To make sure they can talk to each other, both Alice and Bob need two keys: a public key (`pk`) and a private key (`sk`). You can think of each key as a very large number.

Sign(Message, sk) = Signature
Verify(Message, Signature, pk) \in {True/False}

The Signature has a fixed length, say 256 bits. There's only one valid signature for our message among $2^{256}$ options.

RSA is relatively slow. Commonly, we use RSA to establish an ephemeral DH key for a communication. That way, if RSA gets broken, an attacked can't decrypt all of our communications, but they still have to go through them one-by-one.

- VPNs: IKE
- TLS/HTTPS: secure browsing

# Integrity

Ensure integrity (digital fingerprint): SHA-256

## SHA: secure hashing algorithm

Digital signatures, message authentication, etc. they need to be quite quick.

hash function take some string and transform them into a fixed-length binary string. They are pseudo-random: small transformation in the input should produce large changes in the output; they look like garbage. But it's not random at all.

We append the hash at the end of our message, and re-compute the hash? to show that it wasn't modified.

SHA-1: any string as input, string of 160 bits as output.

Ho
