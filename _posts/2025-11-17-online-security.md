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

- __Confidentiality__: ensure that our data and communications are private
- __Authentication__: ensure that we are who we say we are
- __Integrity__: ensure that our data and communications are not tampered with
- __Non-repudiation__: ensure that we cannot deny having authored a message

Let's see how each of them is relevant in our day-to-day online activities.

| Threat                 | Confidentiality | Authentication | Integrity | Non-repudiation |
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

> Throughout this post, I'll be using message to mean data or information. This should bring a more concrete picture, but the contents of this post go well beyond bantering on WhatsApp.

# Confidentiality

Encryption consists on reversibly transforming a message into an (apparently) random message using a secret key. If you have the key, decryption allows you to recover the original information. I will focus on __symmetric__ encryption in this section, i.e., the same key is used for both actions. [Caesar cipher](https://en.wikipedia.org/wiki/Caesar_cipher) is the simplest example. Our key is a number, which indicates how many letters we shift the alphabet by:

| Original letter | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|-----------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Shift by 3      | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | A | B | C |

Then, to encrypt the message "HELLO" with key 3, we shift each letter by 3 positions, resulting in "KHOOR". To decrypt, we simply shift back by 3.

## The cipher: Advanced Encryption Standard

We have come a long way since Caesar cipher. The standard algorithm to encrypt messages these days is the __Advanced Encryption Standard__ (AES). We find it everywhere, in common applications:

- Browsing the internet (TLS)
- Hard drive encryption (e.g., on MacOS)
- WiFi encryption (WPA2 protocol)
- VPNs (IKE)

Same as Caesar cipher, AES is a symmetric encryption algorithm. However, the key is not a numbler, but a long binary string. AES accepts three lengths of keys: 128, 192, or 256 bits. They respectively define the three flavors of AES: AES-128, AES-192, and AES-256, respectively. Longer keys provide more secure encryption, but encryption and decryption become more computationally intensive.

{% details What does it mean for a key to be secure? %}

In short, a key is secure when it cannot be guessed easily. In other words, it's secure when it's _long_, it can only be guessed via _brute force_ and, optionally, testing each guess is _expensive_. For instance, there are $$2^128$$ 128 bit keys.

{% enddetails %}

In a nutshell<d-footnote>It is easy to find detailed explanations around the web, e.g., [here](https://www.geeksforgeeks.org/computer-networks/advanced-encryption-standard-aes/).</d-footnote>, AES stats by decomposing the message into chunks of 16 bytes. Each chunk is processed independently in _rounds_. Each chunk is arranged into a 4-by-4 grid, with with each cell containing 1 byte. One round consists on a pretty complex transformation of the grid, involving dictionary replacements of the cells' contents, shifting rows and columns and, finally, a combination with a key. The key is round-specific, and is derived from the encryption key. The number of rounds depends on the length of the key (10 in AES-128, 12 in AES-192 and 14 in AES-256). To decrypt, the steps are done in reverse order.

## Sharing keys

AES is _everywhere_. It is used gazillions of times every day to encrypt all sorts of communications. However, an obvious problem arises: the need to agree on a common key with which to encrypt the messages. The internet is a very large place, and there are many stops between our computer and (say) our bank's server. Computers need to negotiate a secure key in a way even if someone was listening, they wouldn't be able to get it.

{% details The OG: Diffie-Hellman %}

The Diffie-Hellman algorithm is one of the earliest algorithms to exchange secret keys in public:

1. One of the parties starts by sharing the variables that are used to generate the secret key, two large integers $$x$$ and $$y$$.
1. Each partner generates an intermediate, non-shared key ($$k_1$$ and $$k_2$$). They use that to apply the following transformation:

    $$
    z_i = x^{k_i} \bmod y
    $$

1. The parties share that result publicly ($$z_1$$ and $$z_2$$).
1. The parties leverage each other's intermediate result and their own private key to achieve the same number:

    $$
    \begin{align*}
    z_1^{k_2} \bmod y &= (x^{k_1} \bmod y)^{k_2} \bmod y \\\\
    &= x^{k_1 \cdot k_2} \bmod y \\\\
    &= (x^{k_2} \bmod y)^{k_1} \bmod y \\\\
    &= z_2^{k_1} \bmod y
    \end{align*}
    $$

That's the encryption key.

Diffie-Hellman relies on the fact that modular exponentiation is easy to do, but hard to undo. I.e., if $$x$$ and $$y$$ meet the right conditions, it's hard to recover $$k_1$$ just from $$z_1$$. In fact, the only way is to try all possible intermediate keys untl we stumble upon the right one.

The conditions that Diffie-Hellman needs to work are:

- $$y$$ needs to be a large prime number
- $$x$$ needs to be a _primitive root modulo n_. In other words, $$x^\text{n} \bmod y$$ should produce all positive numbers between 0 and $$y$$.

{% enddetails %}

$$
y^2 = x^3 + ax + b
$$

where $$a$$ and $$b$$ are parameters.

We have a generator $$g$$ which is a point on that curve. Then, we can define additions on the curve. $$2g = g + g$$, which is the result of taking the tangent of the curve at $$g$$, and taking its mirror image. $$3g$$ is the mirror image of the intercept between the curve and the line connecting $$g$$ and $$2g$$. $$4g = 3g + g$$ is the mirror image of the intersection of the curve with the line connecting 3g and g. And so on.

Adding points on an elliptic curve is a way to get points on the curve (apparently) at random. Given a point on the curve, that we know is $$xg$$, x is our secret key.

This is a replacement for Diffie-Hellman. We add a modulo to it.

This is more complicated, but much more efficient mathematically: we can use much shorter keys, and hence to less operations. This is important server-side. The public key is a x, y point, although we can just use x.

# Authentication

Proving identity (prove who you say you are, e.g., to authenticate of exchange keys, credit card payments): ECDSA, Ed25519

The core of the problem is authentication. How can Gmail be sure that the person logging into my email account is really me? Usually, this is done by requesting information that only I should have. Usually, it's one or several of the following:

- Something I know, like a password, a PIN, or the answer to a secret question
- Something I have, like a phone, a hardware token, or a smart card
- Something I am, like a fingerprint, a facial scan, or a voiceprint

A cornerstone of good security is multi-factor authentication (MFA), which requires two or more of these factors to authenticate. That's why many important services these days require you to provide, e.g., both a password and a code sent to your phone.

Of course, the more factors you require, the safer you are. But security comes at the cost of convenience. Maybe you don't want to scan your face and receive an email code to shitpost on Reddit. Maybe you don't have good signal, and would rather keep text messages for the imporant stuff.

## Passwords

- Keep passwords: SHA-256 is too fast. Hence, a hacker could eventually crack your password if your SHA gets leaked. Instead, we use Argon2.

## FIDO2 and passkeys

Passkeys are the implementation of the FIDO2 authentication standard. During setup, your device (browser, phone, security key) creates two keys: a public key (`pk`) and a private key (`sk`).

## RSA: preventing Man-in-the-middle

Imagine there is someone in the middle, Sean. They perform the Diffie-Hellman with both Alice and Bob, while Alice and Bob think they are talking to each other. This ends up in the situation in which both Alice and Bob share a secret with Sean, and none with each other. This is bad. Alice and Bob has no way to know they are not talking to each other.

Diffie-Hellman is designed for two parties to exchange a secret in public, but not when there is someone intercepting the secrets and altering the messages.

To make sure they can talk to each other, both Alice and Bob need two keys: a public key (`pk`) and a private key (`sk`). You can think of each key as a very large number.

Sign(Message, sk) = Signature
Verify(Message, Signature, pk) \in {True/False}

The Signature has a fixed length, say 256 bits. There's only one valid signature for our message among $2^{256}$ options.

RSA is relatively slow. Commonly, we use RSA to establish an ephemeral Diffie-Hellman key for a communication. That way, if RSA gets broken, an attacked can't decrypt all of our communications, but they still have to go through them one-by-one.

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

# TL;DR: Cryptography in practice

## Ensuring confidentiality: AES everywhere!

The first way to ensure confidentiality of **my personal data private** is to encrypt it, in case I lose my devices. If your data is unencrypted, basically anyone can take out the hard drive from your laptop and read its contents.

- MacOS: enable FileVault (`System Settings > Privacy & Security > FileVault`) to encrypt you data using [a variant of AES-256](https://support.apple.com/en-gb/guide/security/sec4c6dc1b6e/web).
- iOS: by default, data is encrypted using AES.
- iCloud: Advanced Data Protection ensures that our data is encrypted _before_ being uploaded to iCloud with a key only you have. This ensures that even if someone gets our iCloud password they can't read it; in theory not even Apple can. Unfortunately, in the UK, His Majesty's Government needs full access to our data, and hence we cannot use this protection.

VPN

- Signal protocol: keeping a conversation secure.
