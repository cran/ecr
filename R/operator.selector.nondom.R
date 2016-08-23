#' @title
#' Non-dominated sorting selector.
#'
#' @description
#' Applies nondominated sorting of the objective and subsequent crowding distance
#' criterion to select a subset of individuals. This is the selector used by the
#' NSGA-II EMOA (see \code{\link{nsga2}}).
#'
#' @return [\code{setOfIndividuals}]
#' @family selectors
#' @export
setupNondomSelector = function() {
  selector = function(fitness, n.select, task, control, opt.state) {
    nondom.layers = doNondominatedSorting(fitness)

    # storage for indizes of selected individuals
    new.pop.idxs = integer()

    # get maximum rank, i.e., the number of domination layers
    max.rank = max(nondom.layers$ranks)

    # get the indizes of points for each domination layer
    idxs.by.rank = lapply(seq(max.rank), function(r) which(nondom.layers$ranks == r))

    # get the number of points in each domination layer ...
    front.len = sapply(idxs.by.rank, length)

    # ... cumulate the number of points of the domination layers ...
    cum.front.len = cumsum(front.len)

    # ... and determine the first domination layer, which does not fit as a whole
    front.first.nonfit = which.first(cum.front.len > n.select)

    if (front.first.nonfit > 1L) {
      # in this case at least one nondominated front can be added
      new.pop.idxs = unlist(idxs.by.rank[1:(front.first.nonfit - 1L)])
    }

    # how many points to select by second criterion, i.e., crowding distance?
    n.diff = n.select - length(new.pop.idxs)

    if (n.diff > 0L) {
      idxs.first.nonfit = idxs.by.rank[[front.first.nonfit]]
      cds = computeCrowdingDistance(fitness[, idxs.first.nonfit, drop = FALSE])
      idxs2 = order(cds, decreasing = TRUE)[1:n.diff]
      new.pop.idxs = c(new.pop.idxs, idxs.first.nonfit[idxs2])
    }

    # merge the stuff and return
    return(new.pop.idxs)
  }

  makeSelector(
    selector = selector,
    name = "NSGA-II survival selector",
    description = "nondominated sorting with potential crowding distance.",
    supported.objectives = "multi-objective"
  )
}