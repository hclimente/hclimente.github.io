---
layout: distill
title: Better day-to-day security
date: 2025-10-10 11:59:00 +0000
description: Cocktail parties make me anxious
tags:
  - linear_algebra
  - statistics
  - machine_learning
giscus_comments: true
related_posts: false
---

# The security-conveniency tradeoff

The core of the problem is authentication. How can Gmail be sure that the person logging into my email account is really me? Usually, this is done by requesting information that only I should have. Usually, it's one or several of the following:

- Something I know, like a password, a PIN, or the answer to a secret question
- Something I have, like a phone, a hardware token, or a smart card
- Something I am, like a fingerprint, a facial scan, or a voiceprint

A cornerstone of good security is multi-factor authentication (MFA), which requires two or more of these factors to authenticate. That's why many important services these days require you to provide, e.g., both a password and a code sent to your phone.

Of course, the more factors you require, the safer you are. But security comes at the cost of convenience. Maybe you don't want to scan your face and receive an email code to shitpost on Reddit. Maybe you don't have good signal, and would rather keep text messages for the imporant stuff.

Threat models I am concerned about:

- Phising attacks
- Phone theft
- Lost devices
