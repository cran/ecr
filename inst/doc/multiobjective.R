## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ----breastCancer, message=FALSE, echo=FALSE----------------------------------
library(ecr)
library(mlr)
library(mlbench)
library(randomForest)
data("BreastCancer")
summary(BreastCancer)

## -----------------------------------------------------------------------------
cancer = BreastCancer[, 2:11]
cancer = cancer[!(rowSums(is.na(cancer)) > 0),]
cancer.features = cancer[, 1:9]
cancer.target = cancer[, 10]

## -----------------------------------------------------------------------------
  fitness.fun = function(ind) {
    ind = as.logical(ind)
    # all features deselected is not a supported solution.
    # Thus, we set the accuracy to 0 and number of features to its maximum.
    if (!any(ind))
      return(c(0, length(ind)))
    # add target column to individual
    task = makeClassifTask(data = cancer[, c(ind, TRUE)],
                           target = "Class",
                           id = "Cancer")
    # Subsampling with 5 iterations and default split ratio 2/3
    rdesc = makeResampleDesc("Subsample", iters = 2)
    # Classification tree
    lrn = makeLearner("classif.randomForest")
    r = do.call(resample, list(lrn, task, rdesc, list(acc), show.info = FALSE))
    measure = r$aggr[[1]]
    nFeatures = sum(ind)
    return(c(measure, nFeatures))
  }

## -----------------------------------------------------------------------------
MU = 5; LAMBDA = 1L; MAX.ITER = 25; N.BITS = ncol(cancer.features);
res = ecr(fitness.fun = fitness.fun,
            n.objectives = 2L,
            minimize = c(FALSE, TRUE),
            representation = "binary",
            n.bits = N.BITS,
            mu = MU,
            lambda = LAMBDA,
            survival.strategy = "plus",
            mutator = setup(mutBitflip, p = 1 / N.BITS),
            p.mut = 0.3,
            p.recomb = 0.7,
            terminators = list(stopOnIters(MAX.ITER)),
            log.pop = TRUE,
            initial.solutions = list(rep(1,N.BITS)))

## ---- message = FALSE, fig.cap = "Pareto front on multi-objective optimization problem", fig.width = 6, fig.height = 4----
plotFront(res$pareto.front)

## ---- echo = TRUE-------------------------------------------------------------
control = initECRControl(fitness.fun, n.objectives = 2L, minimize = c(FALSE, TRUE))
control = registerECROperator(control, "mutate", mutBitflip, p = 0.3)
control = registerECROperator(control, "selectForSurvival", selNondom)

## ---- echo = TRUE-------------------------------------------------------------
  population = genBin(MU, N.BITS)
  fitness = evaluateFitness(control, population)
  archive = initParetoArchive(control)

## ---- echo = TRUE-------------------------------------------------------------
  for (i in seq_len(MAX.ITER)) {
      # sample lambda individuals at random
      idx = sample(1:MU, LAMBDA, replace = TRUE)
      # generate offspring by mutation and evaluate their fitness
      offspring = mutate(control, population[idx], p.mut = 1)
      fitness.o = evaluateFitness(control, offspring)
      # now select the best out of the union of population and offspring
      sel = replaceMuPlusLambda(control, population, offspring, fitness, fitness.o)
      population = sel$population
      fitness = sel$fitness
      updateParetoArchive(archive, population,fitness)
  }

## ---- echo = TRUE, fig.cap = "Pareto front on multi-objective optimization problem", fig.width = 6, fig.height = 4----
pareto.front = getFront(archive)
plotFront(pareto.front)

