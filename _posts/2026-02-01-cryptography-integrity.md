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
