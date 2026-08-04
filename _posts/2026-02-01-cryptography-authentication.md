---
layout: distill
title: Cryptography 101: Authentication
date: 2025-11-17 11:59:00 +0000
description: Cocktail parties make me anxious
tags:
  - algorithms
  - computer_science
  - cryptography
giscus_comments: true
related_posts: false
---

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

- `sign(message_hash: str, sk: key) -> signature: int`. Alice will append a signature to her message. The signature has a fixed length; 256 bytes is a common one. The message is first hashed, then the hash is mapped to one valid signature among all $2^{2048}$ possible ones.
- `verify(message_hash: str, signature: int, pk: key) -> bool`. Upon receiving the message, Bob will hash it, and verify that it originates from Alice using her public key.

RSA is relatively slow. Commonly, we use RSA to establish an ephemeral Diffie-Hellman key for each session. That ensures forward secrecy: if RSA gets broken, an attacker can't decrypt all of our communications, but they still have to go through them one-by-one.

{% details The root of trust: Certificate Authorities %}

You might have noticed that this solution only kicks the problem one level up. How can Bob be sure that the public key comes actually from Alice, and not from Mallory?

Bob has two options. The first one is to get it directly from Alice, which can be impractical. The second one is to rely on the say-so of a third party, a Certificate Authorities (CA).

The CA is someone that we trust. Alice will show her public key to a CA, prove her identity to them, and the CA will sign a Certificate with their own private key, stating that "Public key `0x123...` _definitely_ belongs to Alice". In reality, Alice and Bob don't exchange public keys. _They exchange certificates._ The most powerful certificates are the _root_ certificates: they are the ~150 certificates that come pre-installed in our system and that are not signed by anyone other than themselves. They include big tech, internet companies and different government agencies, among others.

Did we just kick the can one level up? Yes. We are just praying none of these 150 CAs is in cahoots with Mallory.

(A decentralized version of this is the [web of trust](https://en.wikipedia.org/wiki/Web_of_trust).)

{% enddetails %}

# TL;DR: Everyday cryptography

Here are my (subjectively ranked) recommendations.

## Secure your authentication

Someone stealing your authentication credentials can be very damaging. They could steal money from your accounts, send messages under your name to friends and family, get all your emails, delete all the information you store in the cloud, SSH to a server and delete your directories, among many other things. Here's what I do to protect myself from that:

1. Familiarize yourself with [passkeys](#modern-authentication-fido2-and-passkeys), and enable them wherever possible.
1. Enable multi-factor authentication wherever possible.
1. Use a password manager, never repeat the same password twice.
1. Use strong passwords.
1. Secure your critical accounts using a Yubikey.

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
