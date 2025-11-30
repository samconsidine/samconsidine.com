---
title: "Teleop: What I Know"
date: 2024-11-30
author: "Sam Considine"
tags: ["robotics", "teleoperation", "startups"]
---

The whole premise revolves around a simple business case. If I'm deploying robots in the real world, the state of the art is to train the robots in very controlled environments, then roll them out to perform work in that extremely controlled environment.

This is far from optimal. Hardware vendors typically want to ship things as fast as possible to reach scale as large as possible. Then collect data and upgrade the software as models can be trained on that data. This is the purest "flywheel" available in hardware.

Currently, this doesn't really work. Robotics requires a high degree of accuracy in real world use cases, and there's currently no system that can perform real world tasks at the accuracy required to be a good generalist use-case.

If humanoids (and other general purpose robotics) are to be operated in the world, it should be done so with human supervision. Currently that means teleoperation, but teleoperation has its flaws.

To my mind, the ultimate business model would allow hardware manufacturers to deploy teleoperated robots in the real world, then to gather the data and scale autonomy on the fly. To do this, however, two major obstacles must be surmounted:

1. You need people to teleoperate the robots. This is a major bottleneck if you want to scale deployment fast. For a 1-1 robot to teleoperator ratio, you'd need to hire somebody every time you wanted to sell a robot. This is obviously untenable for scaling.
2. Teleoperation software needs to get better. High-latency VR streaming causes motion sickness and poor robot responsiveness. This leads to bad training data. Latency can be optimised away, but if you want to remotely operate a robot in a production setting, there will be some base latency inherent in signal travelling between the operator and robot locations.

The premise of Adamo is to accelerate the scaling of robotics by allowing immediate deployment of robots to collect data in real world settings. This has two components to tackle the two problems mentioned above:

1. A remote teleoperation service that hooks your robot up to teleoperators managed by us, so that robot companies don't have to scale their hiring at the rate that they sell robots.
2. Low-latency and latency-robust teleoperation software that allows for the ergonomic collection of high-quality tele-operation data for imitation learning from a remote location.

We think these goals are achievable with a venture-backed company. The goal of Adamo is to own the whole process of teleoperation, which is the gold standard for robotics data. When people think 'teleoperation' they will think Adamo.
