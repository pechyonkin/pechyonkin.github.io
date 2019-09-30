---
layout: page
title: Portfolio
permalink: /portfolio/
tags: portfolio
---

## Deep Learning

- Developed a [library](https://github.com/pechyonkin/sjaandi) for **deep learning-based visual similarity search, clustering, and image embeddings** [`Python`, `PyTorch`,`fastai`, `Flask`]:
    - While building the library, contributed to several open-source packages (including [fastai](https://github.com/fastai/fastai/commit/10d4e511936ff7bbacc993a5f0452b5737e6f959), a popular open-source deep learning library);
    - To showcase possible use cases of the library built a [Flask application](http://sjaandi.pechyonkin.me) with Docker and deployed it on AWS;
    - Ensured reliability of the library and the project by implementing a continuous integration pipeline in CircleCI using pytest. Also implemented end-to-end browser tests using Selenium;
    - Ported open-source [RasterFairy](https://github.com/pechyonkin/RasterFairy-Py3) library to python3, packaged it and pushed to [PyPI](https://pypi.org/project/rasterfairy-py3/).

- Created an **eye-tracking system** [`Python`, `fastai`, `Flask`]:
    - Built a web app for data collection and labeling, collected data;
    - Trained a deep learning model that predicts which part of laptop screen a user is looking at;
    - Accuracy less than 2 cm;
    - My team productized this technology at a [Beijing Startup Weekend](https://www.eventbank.cn/event/startup-weekend-beijing-21527/) in May 2019 and got third place. 

- Built a **leafy greens classifier** [`Python`, `fastai`, `Docker`]:
    - Scraped more than 8000 images across 29 classes of leafy greens;
    - [Built](https://github.com/pechyonkin/culinary-herbs-classifier) a classifier to tell them apart.

## Traditional Machine Learning

- Made a **predictive model for breast cancer diagnostics** [`Python`, `scikit-learn`]:
    - Performed feature analysis and engineering;
    - Finetuned and validated a variety of models;
    - Best model [achieved](/portfolio/breast-cancer-diagnostics/) 98.1% accuracy, with more than 96% TPR and more than 99% TNR.

- Performed **sentiment analysis of Twitter feed** [`Python`, `R`]:
    - Analyzed 200 gigabytes of Twitter feed data to extract sentiments;
    - Built an index that was predictive of stock market movements.

## Consulting Projects

- Created a proof of concept of **deep learning in industrial defect detection**:
    - Put together a data collection pipeline (industrial cameras and lighting setup), collected and labeled data;
    - Trained a convolutional neural network with performance on par with QA employees (97% accuracy).

- Built an **automated sales pipeline** proof of concept for a client:
    - Performed data cleaning and analysis for auto dealers data all across China, with millions of data points;
    - Estimated customer return dates.

- Delivered two AI product feasibility analyses: 
    - Researched and prepared papers summaries, compiled state of the art results on the topics of **3D pose estimation with deep learning** and **emotion recognition in voice data using deep learning**;
    - As a result, one of the clients proceeded to develop a solution for voice emotion recognition.

## Writing

- [Why Swift may be the next big thing in deep learning](/portfolio/why-swift-for-tensorflow/) - an overview of advantages of Swift in deep learning applications (also [published](https://towardsdatascience.com/why-swift-may-be-the-next-big-thing-in-deep-learning-f3f6a638ca72) on Medium).
- CapsNet Series in four parts ([1](/capsules-1/), [2](/capsules-2/), [3](/capsules-3/), [4](/capsules-4/)) — a popular introduction of the novel neural network architecture. This series was originally published on Medium ([1](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b), [2](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66), [3](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-iii-dynamic-routing-between-capsules-349f6d30418), [4](https://medium.com/@pechyonkin/part-iv-capsnet-architecture-6a64422f7dce)), where it got a lot of attention.
- [Stochastic Weight Averaging](/stochastic-weight-averaging/) — a new way to get state of the art results in deep learning.
- [Deep Learning Vision for Non-Vision Tasks](/deep-learning-vision-non-vision-tasks/) — three case studies about creative application of deep learning vision models to non-vision tasks
- [Key Deep Learning Architectures for Visual Object Recognition](/architectures/) — a series of posts about the most important CNN architectures.

<!--1. [Eye tracking system](/portfolio/eye-tracker/). I built a web app for data collection and then trained a model that predicts which part of laptop screen a user is looking at. [`Python`, `fastai`, `Flask`]
2. Building a [leafy greens classifier](/portfolio/leafy-greens-classifier/). You can try the deployed model [here](https://herbs.onrender.com/). [`Python`, `fastai`, `Docker`]-->

## Autonomous Driving

Below are the projects I did while taking Udacity's Self-Driving Car Nanodegree:

- [Lane detection](/portfolio/lane-finding/) using computer vision techniques [`Python`, `OpenCV`]
- [Traffic sign classification](/portfolio/traffic-signs-classification/) using deep learning [`Python`, `Tensorflow`, `OpenCV`]
- [Behavioral cloning](/portfolio/behavioral-cloning/) using deep learning [`Python`, `Keras`, `OpenCV`]
- [Advanced lane detection](/portfolio/advanced-lane-finding/) with camera calibration. In addition road curvature and vehicle offset relative to the center of line are calculated [`Python`, `OpenCV`]
- Computer vision-based real-time [vehicle detection system](/portfolio/vehicle-detection-cv/) [`Python`, `scikit-learn`, `OpenCV`]

