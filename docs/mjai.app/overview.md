# RiichiLab Overview

Source: https://mjai.app/docs

## Timeline

Matches will be held every weekend. All programs submitted by 11:59 pm JST (02:59 pm UTC) on Friday will be evaluated.

## Submission Format

This competition uses a "submit code" style. Your submission should be a single ZIP archive including a Python program named `bot.py`. You can add data, libraries and binaries to the ZIP archive.

- Make sure that your submission package is smaller than 1GB.
- Your program will run in a Docker container with no network access.

## Code requirements

Your submission package must fulfill the following requirements:

- Your submission package must include a Python program named `bot.py`.
- `bot.py` must accept a seat ID (0~3) as the first program argument.
- `bot.py` must communicate with the game server via standard I/O.
- `bot.py` must be able to win against a tsumogiri bot.
- Your code will be executed in Docker containers with the target platform `linux/amd64` and the following resources: 4G RAM, 2 CPU cores.
- The Docker image `docker.io/smly/mjai-client:v3` including mjai package will be used.

> **Hint: third-party libraries**
> If your code can run on linux/amd64, you can include a compiled library in your submission package.
