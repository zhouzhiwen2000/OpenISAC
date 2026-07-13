---
title: Video and UDP Workflows
description: Payload workflows for UDP traffic and video streaming demonstrations.
---

OpenISAC can carry UDP payloads over the OFDM communication link. Video demos usually feed compressed packets into the same payload path.

## Bring-up order

First validate the PHY with low-rate traffic. Then increase UDP payload rate. Finally add video encoding or decoding tools after link stability is confirmed.

## Debugging

Separate PHY failures from application failures:

- If synchronization is unstable, fix the radio path first.
- If LDPC decode fails, inspect channel and offset estimates.
- If UDP drops occur with stable decode, inspect queues, datagram size, and application pacing.
- If video freezes but UDP arrives, inspect codec and player behavior.

This ordering prevents frontend or application symptoms from being mistaken for radio problems.
