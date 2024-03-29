context("Pareto Archive")

test_that("Pareto archive works well", {
  # dummy init setup
  fun = smoof::makeDTLZ1Function(dimensions = 2L, n.objectives = 2L)
  control = initECRControl(fun)

  # init archive
  max.size = 10L

  # truncation operator based on hypervolume contribution
  truncateByHVContr = function(inds, fitness, max.size, ...) {
    hvcs = computeHVContr(fitness, ...)
    hvcs.ord = order(hvcs, decreasing = TRUE)
    return(list(
      individuals = inds[hvcs.ord[seq_len(max.size)]],
      fitness = fitness[, hvcs.ord[seq_len(max.size)], drop = FALSE])
    )
  }
  archive = initParetoArchive(control, max.size = 10L, trunc.fun = truncateByHVContr)

  # add the following points to the archive (11 - i, i), i = 1, ..., 10
  # each having the same hv contribution
  for (i in seq_len(10)) {
    updateParetoArchive(archive, inds = list(i), fitness = matrix(c(11 - i, i), ncol = 1L))
    expect_equal(getSize(archive), i)
  }

  # now add another elements to the archive with lower HV contributions. These should
  # all being dropped by trunc.fun
  for (i in 2:6) {
    updateParetoArchive(archive, inds = list(9 + i), fitness = matrix(c(11 - i - 0.5, i + 0.5)))
    expect_equal(getSize(archive), max.size)
    expect_true(setequal(unlist(getIndividuals(archive)), 1:10))
    expect_true(is.matrix(getFront(archive)))
  }

  # test that transformFitness works for different optimization directions
  # See https://github.com/jakobbossek/ecr2/issues/126
  fitness = matrix(c(
    1, 1, 2, 2, 3, 3), byrow = FALSE, nrow = 2L)
  dummy.inds = list(1, 2, 3)

  getArchiveSet = function(minimize, inds, fitness) {
    control = initECRControl(identity, n.objectives = 2L, minimize = minimize)
    archive = initParetoArchive(control)
    updateParetoArchive(archive, dummy.inds, fitness)
    getFront(archive)
  }

  minmin = getArchiveSet(c(TRUE, TRUE), dummy.inds, fitness)
  # only (1, 1) is non-dominated
  expect_true(all(as.numeric(minmin) == c(1, 1)))

  maxmax = getArchiveSet(c(FALSE, FALSE), dummy.inds, fitness)
  # only (3, 3) is non-dominated
  expect_true(all(as.numeric(maxmax) == c(3, 3)))

  maxmin = getArchiveSet(c(FALSE, TRUE), dummy.inds, fitness)
  # all three points are non-dominated
  expect_true(all(maxmin == fitness))

  minmax = getArchiveSet(c(TRUE, FALSE), dummy.inds, fitness)
  # same here, i.e., all three points are non-dominated
  expect_true(all(maxmin == fitness))
})
