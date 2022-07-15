# Everything Is RL?

The question: can we reframe all well-defined ML tasks as reinforcement learning
tasks?

I bet we can.

We'll liberally frame "RL" as using a reward signal.

## Tasks

Tasks to test:

- Image classification
- Image segmentation

## Setup

```shell
python -m venv ~/environments/everything-is-rl
source ~/environments/everything-is-rl
pip install -r requirements.txt
python image_classification.py
```
