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

If you have ever read anything about online security and encryption, you'll have quickly felt drowning in a soup of letters: RSA, HTTPS, TLS, SHA, RSA, WPA, FIDO, SHA, PGP, ECDH, AES. Then many of them come with a number appended. Which is sometimes, but not always, a power of 2. It's no wonder that most of us give up before even trying.

In reality, modern cryptography revolves around a few, well-trusted algorithms. Everything else are either wrappers to adapt them to specific applications, or legacy algorithms. These applications try to achieve one or several of these goals:

- __Confidentiality__: ensure that our data and communications are private
- __Authentication__: ensure that we are who we say we are
- __Integrity__: ensure that our data and communications are not tampered with
- __Non-repudiation__: ensure that we cannot deny having authored a message

Let's see how each of them applies to several day-to-day online activities:

| Threat                 | Confidentiality | Authentication | Integrity | Non-repudiation |
|------------------------|-----------------|----------------|-----------|-----------------|
| Private communications | ✓               |                |           |                 |
| Phishing attacks       | ✓               | ✓              | ✓         |                 |
| Theft of devices       | ✓               | ✓              |           |                 |
| Malicious software     |                 | ✓              | ✓         |                 |
| Data tampering         |                 |                | ✓         |                 |
| Online transactions    | ✓               | ✓              |           | ✓               |
| SSH access             | ✓               | ✓              | ✓         |                 |
| `git commit`           |                 |                | ✓         | ✓               |
| `git push`             | ✓               | ✓              | ✓         | ✓               |

In this post, I go over the main algorithms behind each goal, and how I use them to stay safe online. If you don't care about the theory, simply skip to the TL;DR of each section. This won't protect you from either state actors or rubber-hose cryptanalysis, but should be more than enough for 99% of users.

![](https://imgs.xkcd.com/comics/security.png)
<div class="caption">
    From <a href=https://xkcd.com/538>xkcd</a>.
</div>

> Throughout this post, I'll be using message to mean data or information. This should bring a more concrete picture, but the contents of this post go well beyond bantering on WhatsApp.

This post features cherished [Alice and Bob](https://en.wikipedia.org/wiki/Alice_and_Bob). Alice and Bob just want to talk to each other without being snooped in by their evil counterparts, Eve and Mallory.

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

## Sharing keys: Elliptic-curve cryptography

AES is _everywhere_. It is used gazillions of times every day to encrypt all sorts data. But also, to secure __communications__. But how can Alice and Bob agree on a common key in the presence of Eve, who will eavesdrop on each of their conversations? In the old times, Alice and Bob would need to secretly meet in a park to exchange keys in closed envelops, making sure Eve can't get a peek. But in 1977 two researchers, Diffie and Hellman, introduced an algorithm that allowed them to exchange keys in the open, even when Eve could listen to everything they said. This unlocked **public key cryptography** and, ultimately, secure communications over the internet.

At the core of public key cryptography lies a [trapdoor function](https://en.wikipedia.org/wiki/Trapdoor_function), a mathematical function that's easy to do, but very hard to undo. Alice and Bob each apply have their own trapdoor function and, by only sharing its respective outputs, can reach the same mathematical result. And Eve will fall right through the trapdoor, taking her eons to figure out what the function was.

{% details The OG: Diffie-Hellman %}

The Diffie-Hellman algorithm is one of the earliest algorithms to exchange secret keys in public:

1. The protocol determines the variables that will be used to generate the secret key, two large integers $$g$$ and $$p$$.
1. Both Alice and Bob generate an intermediate, non-shared key ($$k_A$$ and $$k_B$$). This is how they define their respective trapdoor functions:

    $$
    z_i = g^{k_i} \bmod p
    $$

1. Alice and Bob share that result publicly ($$z_A$$ and $$z_B$$).
1. Alice and Bob leverage each other's intermediate result and their own private key to achieve the same number:

    $$
    \begin{align*}
    z_A^{k_B} \bmod p &= (g^{k_A} \bmod p)^{k_B} \bmod p \\\\
    &= g^{k_A \cdot k_B} \bmod p \\\\
    &= (g^{k_B} \bmod p)^{k_A} \bmod p \\\\
    &= z_B^{k_A} \bmod p
    \end{align*}
    $$

    That numbers is the encryption key.

Diffie-Hellman relies on the fact that modular exponentiation is easy to do, but hard to undo. I.e., if $$g$$ and $$p$$ meet the right conditions, it's hard to recover $$k_A$$ just from $$z_A$$. In fact, the only way is to try all possible intermediate keys untl we stumble upon the right one.

The conditions that Diffie-Hellman needs to work are:

- $$p$$ needs to be a large prime number
- $$g$$ needs to be a _primitive root modulo n_. In other words, $$g^\text{n} \bmod p$$ should produce all positive numbers between 0 and $$p$$.

{% enddetails %}

Most modern cryptography doesn't revolve around modular exponentiation, but __elliptic curves__. They are the set of points satisfying an equation of the form

$$
y^2 = x^3 + ax + b,
$$

where $$a$$ and $$b$$ are parameters.

We have a generator $$g$$ which is a point on that curve. Then, we can define additions on the curve. $$2g = g + g$$, which is the result of taking the tangent of the curve at $$g$$, and taking its mirror image. $$3g$$ is the mirror image of the intercept between the curve and the line connecting $$g$$ and $$2g$$. $$4g = 3g + g$$ is the mirror image of the intersection of the curve with the line connecting 3g and g. And so on.

Adding points on an elliptic curve is a way to get points on the curve (apparently) at random. Given a point on the curve, that we know is $$xg$$, x is our secret key.

This is a replacement for Diffie-Hellman. We add a modulo to it.

This is more complicated, but much more efficient mathematically: we can use much shorter keys, and hence to less operations. This is important server-side. The public key is a x, y point, although we can just use x.

# Authentication

The core of the problem is authentication. How can Gmail be sure that the person logging into my email account is really me? Usually, this is done by requesting information that only I should have. Usually, it's one or several of the following:

- Something I know, like a password, a PIN, or the answer to a secret question
- Something I have, like a phone, a hardware token, or a smart card
- Something I am, like a fingerprint, a facial scan, or a voiceprint

A cornerstone of good security is multi-factor authentication (MFA), which requires two or more of these factors to authenticate. That's why many important services these days require you to provide, e.g., both a password and a code sent to your phone.

Of course, the more factors you require, the safer you are. But security comes at the cost of convenience. Maybe you don't want to scan your face and receive an email code to shitpost on Reddit. Maybe you don't have good signal, and would rather keep text messages for the imporant stuff.

## Storing passwords: Argon2

- Keep passwords: SHA-256 is too fast. Hence, a hacker could eventually crack your password if your SHA gets leaked. Instead, we use Argon2.

## Modern authentication: FIDO2 and passkeys

Proving identity (prove who you say you are, e.g., to authenticate of exchange keys, credit card payments): ECDSA, Ed25519

Passkeys are the implementation of the FIDO2 authentication standard. During setup, your device (browser, phone, security key) creates two keys: a public key (`pk`) and a private key (`sk`).

https://stephentanner.com/ssh-yubikey.html

# Non-repudiation

## Preventing Man-in-the-middle attacks: RSA

[As before](#ensuring-confidentiality-aes-everywhere), Alice and Bob want to generate a key to encrypt their messages via AES. However, in this case they are not dealing with Eve, but with her evil twin Mallory. As opposed to Eve, Mallory doesn't just eavesdrop; she intercepts and alters the messages.

Hence, while Alice and Bob think they are negotiating their key with each other using [elliptic curves](#sharing-keys-elliptic-curve-cryptography), they are actually negotiating keys with Mallory. This is bad. Alice and Bob have no way to know they are not talking to each other.

To make sure they can talk to each other, they will use the RSA protocol. To that end, both Alice and Bob need two keys: a public key (`pk`) and a private key (`sk`). The RSA provides them with two functions:

- `sign(message: str, sk: key) -> signature: int`. Alice and Bob will append a signature to each of their messages. The signature has a fixed length; 256 bits is a common one. Each message is uniquely mapped to one valid signature among all $2^{256}$ possible ones.
- `verify(message: str, signature: int, pk: key) -> bool`. Upon receiving a message, Alice and Bob will determine its origin using each other's public key.

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

# TL;DR: Cryptography in practice

## Ensuring confidentiality: AES everywhere!

The first way to ensure confidentiality of **my personal data private** is to encrypt it, in case I lose my devices. If your data is unencrypted, basically anyone can take out the hard drive from your laptop and read its contents.

- MacOS: enable FileVault (`System Settings > Privacy & Security > FileVault`) to encrypt you data using [a variant of AES-256](https://support.apple.com/en-gb/guide/security/sec4c6dc1b6e/web).
- iOS: by default, data is encrypted using AES.
- iCloud: Advanced Data Protection ensures that our data is encrypted _before_ being uploaded to iCloud with a key only you have. This ensures that even if someone gets our iCloud password they can't read it; in theory not even Apple can. Unfortunately, in the UK, His Majesty's Government needs full access to our data, and hence we cannot use this protection.

VPN

- Signal protocol: keeping a conversation secure.
