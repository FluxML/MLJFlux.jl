---
title: 'MLJFlux: Deep learning interface to the MLJ toolbox'
tags:
  - Julia
  - Machine learning
  - Deep learning
authors:
  - name: Ayush Shridhar
    orcid: 0000-0003-3550-691X
    affiliation: 1
  - name: Anthony Blaom
    affiliation: 2
affiliations:
 - name: International Institute of Information Technology, Bhubaneswar, India
   index: 1
 - name: Department of Computer Science, University of Auckland
   index: 2
date: 26 March 2021
bibliography: paper.bib
---

# Introduction

We present _MLJFlux.jl_ [@MLJFlux.jl], an interface between the _MLJ_ [@Blaom2020] machine learning toolbox and _Flux.jl_ [@Innes2018] deep learning framework written in the _Julia_ [@Julia-2017] programming language. MLJFlux makes it possible to implement supervised deep learning models while adhering to the MLJ workflow. This means that users familiar with the MLJ design can write their models in Flux with a few slight modifications and perform all tasks provided by the MLJ model spec which includes evaluation, fitting on a dataset, hyperparameter tuning or plotting. The interface also provides options to train the model on different hardware and warm start the model after changing some specific hyper-parameters.

Julia aims to solve the _"two language problem"_ in scientific computing, where high-level languages such as Python or Ruby are easy to use but often slow and low-level languages are fast but difficult to use. Using _just-in-time_ compilation and _multiple dispatch_, Julia makes the best of both worlds by matching the performance of low-level languages [@JuliaBenchmarks] while also being adaptable.

While there has been significant work towards enabling the creation and delivery of deep learning models with _FastAI.jl_ [@FastAI.jl], the focus has been extensively on neural network paradigms. It provides convenience functions for loading data, modeling and tuning but the process still involves writing the preprocessing pipelines, loss functions, optimizers and other hyper-parameters. MLJFlux.jl tries to remove the need for this boilerplate code by leveraging the MLJ design which makes it ideal for basic prototyping and experimentation.

# Statement of Need

While MLJ supports multiple statistical models, it lacks support for any kind of deep learning model. MLJFlux adds support for this by interfacing MLJ with the Flux.jl deep learning framework. Converting a Flux model into an MLJ spec can be done by wrapping it in the appropriate MLJFlux container and specifying other hyper-parameters such as the loss function, optimizer, epochs, and so on. This can now be used like any other MLJ model. It can be further be wrapped in an MLJ _machine_, which in MLJ terminology is a way of binding a container for hyper-parameters (i.e. MLJ _model_) with data, and then any functionalities that work with MLJ machines can be invoked. MLJFlux also gives the ability to warm start the training upon changing certain model hyper-parameters.

MLJFlux provides four containers that we can enclose our Flux model in. Each model is derived from either `MLJModelInterface.Probabilistic` or `MLJModelInterface.Deterministic` to follow the MLJ design, depending on the type of task (regression or prediction). At the core of each wrapper is a _builder_ attribute that specifies the Flux model. The _builder_ defines instructions on how to build the Flux model once we've seen the data.

MLJFlux has been written with ease of use in mind. The core idea is to allow rapid modeling, tuning and visualization of deep learning models via Flux.jl by reusing the already mature and efficient functionalities offered by MLJ.

# References