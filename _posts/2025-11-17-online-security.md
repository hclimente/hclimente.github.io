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

If you've ever read anything about online security and privacy, you'll have quickly felt drowning in a soup of letters: RSA, HTTPS, SHA, RSA, WPA, TLS, FIDO, PGP, ECDH, AES. Many of them come with a number appended. Which is sometimes, but not always, a power of 2. No wonder most of us give up before even trying.

In reality, modern cryptography is built around a few, well-trusted algorithms. Everything else are wrappers to adapt them to specific applications (or legacy algorithms we should ditch as soon as possible!). Each application tries to address one or more of these goals:

- __Confidentiality__: ensure that our data and communications are private
- __Authentication__: ensure that we are who we say we are
- __Integrity__: ensure that our data and communications are not tampered with
- __Non-repudiation__: ensure that we cannot deny having authored a message

Let's see how each of them applies to several day-to-day online activities:

| Scenario              | Confidentiality | Authentication | Integrity | Non-repudiation |
|-----------------------|-----------------|----------------|-----------|-----------------|
| Web browsing          | ✓               |                |           |                 |
| Login/email           | ✓               | ✓              | ✓         |                 |
| Local storage         | ✓               | ✓              |           |                 |
| Software downloads    |                 | ✓              | ✓         |                 |
| File transfers        |                 |                | ✓         |                 |
| Online payments       | ✓               | ✓              |           | ✓               |
| SSH access            | ✓               | ✓              | ✓         |                 |
| `git commit`          |                 |                | ✓         | ✓               |
| `git push`            | ✓               | ✓              | ✓         | ✓               |

In this post, I go over the main algorithms behind each goal, and how I use them to stay safe. If you don't care about the theory, simply skip to the TL;DR of each section. This won't protect you from either state actors or _wrench cryptanalysis_, but should be more than enough for 99% of us.

![](https://imgs.xkcd.com/comics/security.png)
<div class="caption" align="center">
    From <a href="https://xkcd.com/538">xkcd</a>.
</div>

> Throughout this post, I'll be using _message_ to mean _data_ or _information_. This should bring a more concrete picture, but the contents of this post go well beyond bantering on WhatsApp.

Our heroes in this story will be [Alice and Bob](https://en.wikipedia.org/wiki/Alice_and_Bob). Alice and Bob just want to talk to each other without being snooped in by their evil counterparts, Eve and Mallory.

# Prelude: finite field arithmetic

Cryptography works with integers, rather than with real numbers. One advantage is that integers do not suffer from rounding errors. Cryptography often consists on performing a fixed series of calculations to arrive to the same conclusion. When each step incurs in a precision cost, that is no longer guaranteed.

To make the maths work, they rely on __finite fields__, also known as Galois fields. Let's break that down.

First, the field is __finite__, that is, it operates on a bounded set of items. In our case a set of integers. The size of the set is called the _order_ of the field.

Second, it _is_ a __field__, that is, it has 4 binary operators (multiplication, addition, subtraction and division) satisfying the field axioms. An important one is that the result of the operation must also be in the field.

The classical example is the set of integers modulo $$p$$, with $$p$$ being a prime:

$$
\mathbb{Z} / p \mathbb{Z} = \{ \bar{a}_p | a \in \mathbb{Z} \}
$$

where $$\bar{a}_p$$ represents the entire set of integers that produce the same remainder as $$a$$ when divided by $$p$$.

# Confidentiality

| __Key algorithms__ | AES, elliptic curve cryptography |
| __Key protocols__  | TLS/HTTPS, WPA3, IKE             |

The main tool to ensure that our communications remain private is __encryption__. Encryption consists on reversibly transforming a message into an (apparently) random message using an encyption key. If you have the decryption key, decryption allows you to recover the original information. This section focuses on __symmetric__ encryption, i.e., the same key is used for both actions. The [Caesar cipher](https://en.wikipedia.org/wiki/Caesar_cipher) is the simplest example. Our key is a number, which indicates how many letters we shift the alphabet by:

| Original letter | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|-----------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Shift by 3      | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | A | B | C |

To encrypt the message "HELLO" with _key 3_, we shift each letter by 3 positions, resulting in "KHOOR". To decrypt, we simply shift back by 3.

## The cipher: Advanced Encryption Standard

Given a long enough message, breaking the Caesar cipher is easy, since we know the expected probability of each word in the English language. Or we can simply brute-force all 26 possibilities.

Luckily the field is much more sophisticated now. The standard encryption algorithm is the __Advanced Encryption Standard__ (AES). Same as Caesar cipher, AES is a symmetric encryption algorithm. However, the key is not a number, but a long binary string. AES accepts three lengths of keys: 128, 192, or 256 bits. They respectively define the three flavors of AES: AES-128, AES-192, and AES-256. Longer keys provide more secure encryption, but encryption and decryption become more computationally intensive.

{% details What does it mean for a key to be secure? %}

In short, a key is secure when it cannot be guessed easily. In other words, it's secure when it's _long_, it can only be guessed via _brute force_ and, optionally, testing each guess is _expensive_. For instance, there are $$2^128$$ 128 bit keys. In contrast, Caesar cipher has only 26 possible keys.

{% enddetails %}

AES is pretty convoluted algorithm, and I find it a bit uninteresting. Kind of like an algorithm to shuffle cards reproducibly; interesting and lucrative applications, boring to watch. Luckily for me there are many good, detailed explanations around the web I can point you to (e.g., [here](https://www.geeksforgeeks.org/computer-networks/advanced-encryption-standard-aes/)).

The TL;DR is this: AES starts by decomposing the message into chunks of 16 bytes, which are arranged into a 4-by-4 grid, with each cell containing 1 byte. Each chunk is then processed independently in _rounds_. One round consists on a pretty complex transformation of the grid, involving dictionary replacements of the cells' contents, shifting rows and columns and, finally, a combination with a key. The key is round-specific, and is derived from the encryption key. The number of rounds depends on the length of the key (10 in AES-128, 12 in AES-192 and 14 in AES-256). Decrypting the data consists on performing the steps in reverse order.

## Sharing keys: public key cryptography

AES is _everywhere_, and it is used gazillions of times every day to encrypt all hard drives (e.g., on MacOS).

But what about securing _communications_? How can Alice and Bob agree on a common key in the presence of Eve, who will eavesdrop on each of their conversations? In the old times, Alice and Bob would meet in a park and exchange keys in closed envelopes, making sure Eve can't get a peek. But in 1976 two researchers, Diffie and Hellman, introduced an algorithm that allowed them to agree on a key in the open, even when Eve could listen to everything they said to each other. This unlocked __public key cryptography__ and, ultimately, secure communications over the internet, like browsing the internet (implemented in TLS/HTTPS), securing our WiFi (WPA3), or using a VPN (IKE).

### Trapdoor functions

At the core of public key cryptography lies a [trapdoor function](https://en.wikipedia.org/wiki/Trapdoor_function), a mathematical function that's easy to do, but very hard to undo. Alice and Bob each apply have their own trapdoor function and, by only sharing its respective outputs, can reach the same mathematical result. And Eve will fall right through the trapdoor, taking her eons to figure out what the functions were.

A classic example of a trapdoor function is __modular exponentiation__, used by Diffie-Hellman.

$$
g^{k} \bmod p
$$

If tell I you the second hand of my clock ($$\text{Current time} \bmod 60$$) is pointing at 32 ($$g = 32$$) seconds, and that I raise that number to the 4th power ($$k = 4$$), you will have no issues computing that it will end up pointing at

$$
32^{4} \bmod 60 = 16.
$$

But if only I tell you I started at 32s and ended at 16, and ask you what $$k$$, that's the __discrete logarithm problem__, and it is much harder. In fact, you'll only be able to solve it by enumerating all possibilities:

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

Now, it turns out 60 is not a great choice for $$p$$. The space of outcomes encompasses, at most, the 60 ticks of the clock. That means that we just need to enumerate 60 possibilities to identify which number $$k$$ is a multiple of, reducing our search space by a factor of 60. The larger $$p$$ is, the harder this problem becomes: we want $$\boldsymbol{p}$$ __to be astronomically large__; Diffie-Hellman makes it at least 2048 bits long.

But, given that we are doing modulo 60, 32 is not a good choice for $$g$$ either. Out of the 60 outcomes $$\operatorname{mod} 60$$ offers, the powers of 32 modulo 60 occupy only 4: the solution to $$k=1$$ is the same as to $$k=5$$: $$32^{5} \bmod 60 = 32$$. Hence, we quickly shrink our space of possibilities by a factor of 15: only multiples of 4 could produce a remainder of 16. We want the opposite: all options between 0 and $$p$$ should be possible (i.e., we want $$g$$ to be a __primitive root modulo__ $$\boldsymbol{p}$$.) While there are better choices than 32, [there are no primitive roots modulo 60](https://en.wikipedia.org/wiki/Primitive_root_modulo_n#:~:text=A%20primitive%20root%20exists%20if%20and,odd%20prime%20and%20k%20%3E%200.), and hence we should ditch it altogether. A better choice would be $$g=5$$ and $$p = 6$$, since 5 is a primitive root modulo 6. An even better choice would be to pick a massive, prime $$p$$, and a small $$g$$ that is a primitive root modulo $$p$$.

### The OG: the Diffie-Hellman algorithm

Now that we understand its trapdoor function, let's go back to Diffie-Hellman. Alice and Bob want to communicate privately with each other. To that end, they will encrypt each of their messages using AES-256. However, they haven't agreed on an encryption key yet, and Eve is there listening to everything they say to each other. (Not too unlike anytime we access a website on public WiFi.)

This is how Diffie and Hellman solved this issue:

1. Alice and Bob will first establish a communication using a pre-agreed protocol. The protocol determines the two integers $$g$$ and $$p$$ that will provide a good trapdoor $$f$$:

    $$
    f(k) = g^{k} \bmod p
    $$

1. Both Alice and Bob generate a secret large integer ($$k_A$$ and $$k_B$$, respectively) that they never share with each other (and hence with Eve). They will use it to define their respective trapdoor functions, and apply it to $$g$$ and $$p$$:

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

### Elliptic curve cryptography

A crucial drawback of Diffie-Hellman is the computational burden of integer arithmetic on astronomically large $$p$$. This gets really expensive at scale! Consider the burdent on servers that run all these computations for millions of concurrent connections. Luckily modern cryptography has mostly moved past Diffie-Hellman's modular exponentiation and into __elliptic curves__, which offer comparable security with much smaller key sizes.

Elliptic curves are the set of points satisfying an equation of the form

$$
y^2 = x^3 + ax + b,
$$

where $$a$$ and $$b$$ are parameters. They are defined over a finite field.

#### The trapdoor

Elliptic curves are defined over a finite field of whole numbers. However, let's set that aside for now, and develop our intuitions on the continuous case.

Elliptic curves cryptography relies on __elliptic curve point addition and multiplication__. Elliptic curve __point addition__ of a point ($$P + Q$$) consists on taking the line connecting $$P$$ and $$Q$$, intersecting it with the curve itself and taking the mirror image.

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_addition.gif" class="img-fluid" %}
<div class="caption" align="center">
    <b>Elliptic curve point addition.</b>
</div>

Then, elliptic curve __point multiplication__ consists on doing that over and over to the same point. In the absence of another point to compute a line, we will take the tangent of the curve at $$P$$ to compute $$2P$$. Then, we simply keep adding $$P$$ to the resulting number to compute higher multiples.

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_multiplication.gif" class="img-fluid" %}

<div class="caption" align="center">
    <b>Elliptic curve point multiplication.</b>
</div>

__Our trapdoor is elliptic curve point multiplication.__ Given a _generator_ point $$P$$ and a factor $$k$$, it is easy to compute $$kP$$. However, given $$kP$$ and $$P$$, it is really hard to guess the value of $$k$$. $$P$$ is our public key, and $$k$$ is our secret key.

#### Finite fields

Working with real numbers is hard for computers, since precision is lost. That's why cryptography prefers to work with integers. Remember I said elliptic curves are defined over a finite field? Let's revisit that.

We can define a curve as the set of solutions to

$$
y^2 \equiv x^3 + ax + b \pmod{p}
$$

where $$p$$ is a prime, defined over $$\mathbb{Z} / p \mathbb{Z}$$. It would be hard to see why this is a _curve_, although we should, by analogy to the continuous case:

{% include figure.liquid path="assets/python/2025-11-17-online-security/img/elliptic_curve_finite_field.png" class="img-fluid" %}

<div class="caption" align="center">
    <b>Elliptic curve on a finite field.</b>
</div>

This is a replacement for Diffie-Hellman. We add a modulo to it.

This is more complicated, but much more efficient mathematically: we can use much shorter keys, and hence to less operations. This is important server-side. The public key is a x, y point, although we can just use x.

# Authentication

| __Key algorithms__ | Ed25519         |
| __Key protocols__  | FIDO2, Webauthn |

The core of the problem is authentication. How can Gmail be sure that the person logging into my email account is really me? Usually, this is done by requesting information that only I should have. Usually, it's one or several of the following:

- Something I know, like a password, a PIN, or the answer to a secret question
- Something I have, like a phone, a hardware token, or a smart card
- Something I am, like a fingerprint, a facial scan, or a voiceprint

A cornerstone of good security is multi-factor authentication (MFA), which requires two or more of these factors to authenticate. That's why many important services these days require you to provide, e.g., both a password and a code sent to your phone.

Of course, the more factors you require, the safer you are. But security comes at the cost of convenience. Maybe you don't want to scan your face and receive an email code to shitpost on Reddit. Maybe you don't have good signal, and would rather keep text messages for the imporant stuff.

{% details The downsides of a fast algorithm %}

Passwords should never be stored in plain text. Instead, the its hash should be stored. Then, authenticating requires the user to prove they know the password that can produce that hash. And, if and when the database gets hacked, an attacker cannot fetch you login credentials. While SHA-256 is a common hashing algorithm, it has a downside: it is _fast_; too fast. So much than it becomes feasible for an attacker to brute force your password. That's why we instead use [Argon2](https://en.wikipedia.org/wiki/Argon2) to compute passwords hashes, which is designed to be resistant to bruteforce attacks.

{% enddetails %}

## Modern authentication: FIDO2 and passkeys

While we are all familiar with passwords and TOTPs, industry is recently pushing for passwordless authentication via **passkeys**. They rely on private key cryptography and blend multiple factors into one.

Proving identity (prove who you say you are, e.g., to authenticate of exchange keys, credit card payments): ECDSA, Ed25519

Passkeys are the implementation of the FIDO2 authentication standard. During setup, your device (browser, phone, security key) creates two keys: a public key (`pk`) and a private key (`sk`).

https://stephentanner.com/ssh-yubikey.html

## Preventing Man-in-the-middle attacks: RSA

[As before](#ensuring-confidentiality-aes-everywhere), Alice and Bob want to generate a key to encrypt their messages via AES. However, in this case they are not dealing with Eve, but with her evil twin Mallory. As opposed to Eve, Mallory doesn't just eavesdrop; she intercepts and alters the messages.

Hence, while Alice and Bob think they are securely agreeing on a key using [elliptic curves](#sharing-keys-elliptic-curve-cryptography), they are actually negotiating keys with Mallory. This is bad. Alice and Bob have no way to know they are not talking to each other!

To make sure they can talk to each other, they will use the RSA protocol. To that end, both Alice and Bob need two keys: a public key (`pk`) and a private key (`sk`). The RSA provides them with two functions:

- `sign(message: str, sk: key) -> signature: int`. Alice and Bob will append a signature to each of their messages. The signature has a fixed length; 256 bytes is a common one. Each message is uniquely mapped to one valid signature among all $2^{2048}$ possible ones.
- `verify(message: str, signature: int, pk: key) -> bool`. Upon receiving a message, Alice and Bob will determine its origin using each other's public key.

RSA is relatively slow. Commonly, we use RSA to establish an ephemeral Diffie-Hellman key for each session. That ensures forward secrecy: if RSA gets broken, an attacker can't decrypt all of our communications, but they still have to go through them one-by-one.

# Non-repudiation



# Integrity

Ensure integrity (digital fingerprint): SHA-256

## SHA: secure hashing algorithm

Digital signatures, message authentication, etc. they need to be quite quick.

hash function take some string and transform them into a fixed-length binary string. They are pseudo-random: small transformation in the input should produce large changes in the output; they look like garbage. But it's not random at all.

We append the hash at the end of our message, and re-compute the hash? to show that it wasn't modified.

SHA-1: any string as input, string of 160 bits as output.

# TL;DR: Everyday cryptography

Here are my (subjectively ranked) recommendations.

## Secure your authentication

Someone stealing your authentication credentials can be very damaging. They could steal money from your accounts, send messages under your name to friends and family, get all your emails, delete all the information you store in the cloud, SSH to a server and delete your directories, among many other things. Here's what I do to protect myself from that:

1. Familiarize yourself with [passkeys](#modern-authentication-fido2-and-passkeys), and enable them wherever possible.
1. Enable multi-factor authentication wherever possible.
1. Use a password manager, never repeat the same password twice.
1. Use strong passwords.
1. Secure your critical accounts using a Yubikey.

## Encrypt your data!

The first way to ensure confidentiality of my personal data private is to encrypt it, in case I lose my devices. If your data is unencrypted, basically anyone can take out the hard drive from your laptop and read its contents. In the Apple ecosystem, this involves:

- MacOS: enable FileVault (`System Settings > Privacy & Security > FileVault`) to encrypt you data using [a variant of AES-256](https://support.apple.com/en-gb/guide/security/sec4c6dc1b6e/web).
- iOS: by default, data is encrypted using AES.
- iCloud: Advanced Data Protection ensures that our data is encrypted _before_ being uploaded to iCloud with a key only you have. This ensures that even if someone gets our iCloud password they can't read it; in theory not even Apple can.

> Unfortunately, in the UK, His Majesty's Government needs full access to our data, and hence we cannot use this protection.

## Encrypt your communications

VPN

- Signal protocol: keeping a conversation secure.

## Up your git game

### Signing your commits

```bash
# e.g.
GIT_EMAIL="noreply@hclimente.eu"
# assuming your key is the first one in the agent
PUBLIC_KEY=$(ssh-add -L | head -n1)

# set git settings
git config --global gpg.format ssh
git config --global user.signingKey "key::$PUBLIC_KEY"
git config --global commit.gpgsign true
git config --global tag.gpgsign true

mkdir -p ~/.config/git
echo $GIT_EMAIL $PUBLIC_KEY >~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile "$HOME/.config/git/allowed_signers"
```
