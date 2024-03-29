context("test logger")

test_that("logger keeps track the right way in single-objective case", {
  # dummy control
  control = initECRControl(function(x) sum(x), minimize = TRUE, n.objectives = 1L)

  evals.per.iter = 10L
  n.iters = 20L
  log.stats = list(fitness = list("min", "median", "ncol", "myRange" = function(x) max(x) - min(x)))
  log.extras = c(double = "numeric", number = "integer", mutator = "character")

  log = initLogger(control,
    log.stats = log.stats,
    log.extras = log.extras,
    log.pop = TRUE,
    init.size = 10L)

  for (i in seq.int(n.iters)) {
    population = genBin(10L, evals.per.iter)
    #population = replicate(sample(c(0, 1), 10L, replace = TRUE), n = evals.per.iter, simplify = FALSE)
    fitness = matrix(sapply(population, sum), nrow = 1L)
    cat(i)
    for (j in seq_along(population)) {
      attr(population[[j]], "fitness") = fitness[, j]
    }
    extras = list(double = runif(1), number = sample(1:10, 1L), mutator = c("mut1", "mut2")[sample(1:2, 1L)])
    updateLogger(log, population, n.evals = evals.per.iter, extras = extras)
    expect_true(nrow(getStatistics(log)) == i)
  }

  # now check that stuff
  stats = getStatistics(log)

  # check that stats is a data.frame
  assertDataFrame(stats, nrows = n.iters, ncols = 9L,
    any.missing = FALSE, all.missing = FALSE)

  # check for logged stats
  expected.stats = c("gen", "time.passed", "fitness.min", "fitness.median", "fitness.ncol", "fitness.myRange", "double", "number", "mutator")
  expect_set_equal(colnames(stats), expected.stats)

  # check stats df to ggplot-friendly df helpers
  stats.gg = toGG(stats)
  expect_set_equal(colnames(stats.gg), c("gen", "stat", "value"))
  expect_set_equal(as.character(unique(stats.gg$stat)), expected.stats[-1L])

  # check plotting
  pl = plotStatistics(stats)
  expect_class(pl, "ggplot")
  pl = plotStatistics(log)
  expect_class(pl, "ggplot")

  # now with dropping of stats
  stats.gg = toGG(log, drop.stats = c("fitness.ncol", "fitness.myRange"))
  expect_set_equal(colnames(stats.gg), c("gen", "stat", "value"))
  expect_set_equal(as.character(unique(stats.gg$stat)), expected.stats[-c(1, 5, 6)])

  # check logged populations
  pops  = getPopulations(log)
  expect_length(pops, n.iters)
  expect_true(!all(sapply(pops, is.null)))
})

