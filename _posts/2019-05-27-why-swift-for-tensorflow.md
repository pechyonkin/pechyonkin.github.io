---
layout:     post
title:      Why Swift May Be the Next Big Thing in Deep Learning
date:       2019-05-27
summary:    If you are into deep learning, then Swift is a language you should probably start learning. Why? Learn in this post.
permalink:	/portfolio/:title/
use_math:	false
subscription-form: true
---

## Introduction

If you are into programming, when you hear Swift, you will probably think about app development for iOS or MacOS. If you’re into deep learning, then you must have heard about [Swift for Tensorflow](https://www.tensorflow.org/swift/) (abbreviated as S4TF). Then, you can ask yourself: “Why would Google create a version of TensorFlow for Swift? There are already versions for Python and C++; why add another language?” In this post, I will try to answer this question and outline the reasons why you should carefully follow  S4TF as well as the Swift language itself. The goal of this post is not to give very detailed explanations but to provide a general overview with plenty of links so that you can go and dig deeper if you get interested.

## 🧠 Swift Has Strong Support Behind It
Swift was created by [Chris Lattner](https://en.wikipedia.org/wiki/Chris_Lattner) when he was working at Apple. Now, Chris Lattner works at [Google Brain](https://ai.google/research/teams/brain), one of the best artificial intelligence research teams in the world. The fact that the creator of the Swift language now works at a lab researching deep learning should tell you that this is a serious project. 

Some time ago, people at Google realized that even though Python is an excellent language, it has many limitations that are hard to overcome. A new language was needed for TensorFlow, and after long deliberation, Swift was chosen as a candidate. I will not go into details here, but [there](https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md) is a document that describes drawbacks of Python and what other languages were considered and how eventually it was narrowed down to Swift.

## 💪 Swift for TensorFlow is Much More Than Just a Library
Swift for TensorFlow is not just TF for another language. It is essentially another branch (in [git terms](https://git-scm.com/book/en/v1/Git-Branching-What-a-Branch-Is)) of the Swift language itself. This means that S4TF is not a library; it is a language in its own right, with features built into it that support all functionality needed for TensorFlow. For example, S4TF has very powerful [automatic differentiation](https://github.com/tensorflow/swift/blob/master/docs/AutomaticDifferentiation.md) system in it, which is one of the foundations of deep learning needed for calculating gradients. Contrast this with Python, where automatic differentiation is not a core component of the language. Some of the features developed initially as part of S4TF were later integrated into the Swift language itself.

## ⚡️ Swift is Fast
When I first learned that Swift runs as fast as C code, I was astonished. I knew that C was highly optimized and allowed to achieve very high speed, but that comes at the cost of micro-managing memory, which leads to C being not memory safe). Besides, C is not a language that is very easy to learn. 

Now, Swift [runs as fast as C](https://www.fast.ai/2019/01/10/swift-numerics/) in numerical computation, *and* it doesn’t have memory safety issues, *and* it is much easier to learn. LLVM compiler behind Swift is very powerful and has very efficient optimizations that will ensure your code will run very fast.

## 📦 You Can Use Python, C and C++ Code in Swift
Since Swift for machine learning is at a very early stage of its life, this means there are not many machine learning libraries for Swift. You shouldn’t worry about it too much, because Swift has amazing [Python interoperability](https://github.com/tensorflow/swift/blob/master/docs/PythonInteroperability.md). You simply import any Python library in Swift, and it just works. Similarly, you can [import C and C++ libraries](https://oleb.net/blog/2017/12/importing-c-library-into-swift/) into Swift (for C++, you need to make sure that header files are written in plain C, without C++ features).

To summarize, if you need specific functionality, but it is not implemented in Swift yet, you can import corresponding Python, C or C++ package. Impressive!

## ⚙️ Swift Can Go Very Low-Level
If you ever used TensorFlow, most likely you did it through a Python package. Underneath the hood, Python version of TensorFlow library has C code underneath. So when you call any function in TensorFlow, at some level you hit some C code. This means there is a limit of how low you can go trying to inspect the source code. For example, if you want to see how convolutions are implemented, you won’t be able to see Python code for that, because that’s implemented in C. 

In Swift, that is different. Chris Lattner called Swift  “[syntactic sugar for LLVM](https://www.fast.ai/2019/03/06/fastai-swift/) [assembly language]”. This means that in essence, Swift sits very close to the hardware, and there are no other layers of code written in C in between. This also means that the Swift code is very fast, as was described above. It all leads to you as developer being able to inspect code from a very high level to a very low level, without the need to go into C.

## 📈 What’s Next
Swift is only one part of the innovation in deep learning happening at Google. There is another component that is very closely related: [MLIR](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d), which stands for Multi-Level Intermediate Representation. MLIR will be Google’s unifying compiler infrastructure, allowing to write code in Swift (or any other supported language) and compile it to any supported hardware. Currently, there are a plethora of compilers for different target hardware, but MLIR will change that, allowing not only for code reusability but also for writing custom low-level components of the compiler. It will also allow researchers to apply machine learning to optimize low-level algorithms:

> While MLIR acts as a compiler for ML, we also see it enabling the use of machine learning techniques within compilers as well! This is particularly important as engineers developing numerical libraries do not scale at the same rate as the diversification of ML models or hardware.

Imagine being able to use deep learning to help optimize low-level memory tiling algorithms on data (a task similar to what [Halide](https://www.youtube.com/watch?v=3uiEyEKji0M) is trying to accomplish). Moreover, this is only the beginning and other creative applications of machine learning in compilers away!

## Summary
If you are into deep learning, then Swift is a language you should probably start learning. It brings many advantages as compared to Python. Google is investing heavily into making Swift a key component of its TensorFlow ML infrastructure, and it is very likely Swift will become *the* language of deep learning. So, starting to get involved with Swift early will give you first-mover advantage.

## Links for Further Exploration
* [fast.ai Embracing Swift for Deep Learning · fast.ai](https://www.fast.ai/2019/03/06/fastai-swift/)
* [Understanding Swift for TensorFlow](https://towardsdatascience.com/machine-learning-with-swift-for-tensorflow-9167df128912)

**Note**: this article is also available [on Medium](https://towardsdatascience.com/why-swift-may-be-the-next-big-thing-in-deep-learning-f3f6a638ca72).