#' @title
#' Average Hausdorff Distance computation.
#'
#' @description
#' Computes the average Hausdroff distance measure between two point sets.
#'
#' @param A [\code{matrix}]\cr
#'   First point set (each column corresponds to a point).
#' @param B [\code{matrix}]\cr
#'   Second point set (each column corresponds to a point).
#' @param p [\code{numeric(1)}]\cr
#'   Parameter p of the average Hausdoff metrix. Default is 1. See the description
#'   for details.
#' @template arg_asemoa_dist_fun
#' @return [\code{numeric(1)}] Average Hausdorff distance of sets \code{A} and \code{B}.
#' @export
computeAverageHausdorffDistance = function(A, B, p = 1, dist.fun = computeEuclideanDistance) {
  # sanity check imput
  assertMatrix(A, mode = "numeric", any.missing = FALSE, all.missing = FALSE)
  assertMatrix(B, mode = "numeric", any.missing = FALSE, all.missing = FALSE)
  if (nrow(A) != nrow(B)) {
    stopf("Sets A and B need to have the same dimensionality.")
  }
  assertNumber(p, lower = 0.0001, na.ok = FALSE)

  # ac
  GD = computeGenerationalDistance(A, B, p, dist.fun)
  IGD = computeInvertedGenerationalDistance(A, B, p, dist.fun)
  delta = max(GD, IGD)
  return(delta)
}

computeEuclideanDistance = function(x) {
  sqrt(sum(x^2))
}

#' @title
#' Computes Generational Distance.
#'
#' @description
#' Helper to compute the Generational Distance (GD) between two sets of points.
#'
#' @param A [\code{matrix}]\cr
#'   First point set (each row corresponds to a point).
#' @param B [\code{matrix}]\cr
#'   Second point set (each row corresponds to a point).
#' @param p [\code{numeric(1)}]\cr
#'   Parameter p of the average Hausdoff metrix. Default is 1. See the description
#'   for details.
#' @template arg_asemoa_dist_fun
#' @return [\code{numeric(1)}]
#' @export
computeGenerationalDistance = function(A, B, p = 1, dist.fun = computeEuclideanDistance) {
  assertMatrix(A, mode = "numeric", any.missing = FALSE, all.missing = FALSE)
  assertMatrix(B, mode = "numeric", any.missing = FALSE, all.missing = FALSE)
  if (nrow(A) != nrow(B)) {
    stopf("Sets A and B need to have the same dimensionality.")
  }
  assertNumber(p, lower = 0.0001, na.ok = FALSE)

  # compute distance of each point from A to the point set B
  dists = apply(A, 2L, function(a) computeDistanceFromPointToSetOfPoints(a, B))
  GD = mean(dists^p)^(1 / p)
  return(GD)
}

#' @title
#' Computes Inverted Generational Distance.
#'
#' @description
#' Helper to compute the Inverted Generational Distance (IGD) between two sets
#' of points.
#'
#' @param A [\code{matrix}]\cr
#'   First point set (each row corresponds to a point).
#' @param B [\code{matrix}]\cr
#'   Second point set (each row corresponds to a point).
#' @param p [\code{numeric(1)}]\cr
#'   Parameter p of the average Hausdoff metrix. Default is 1. See the description
#'   for details.
#' @template arg_asemoa_dist_fun
#' @return [\code{numeric(1)}]
#' @export
computeInvertedGenerationalDistance = function(A, B, p = 1, dist.fun = computeEuclideanDistance) {
  return(computeGenerationalDistance(B, A, p))
}

#' @title
#' Computes distance between a single point and set of points.
#'
#' @description
#' Helper to compute distance between a single point and a point set.
#'
#' @param a [\code{numeric(1)}]\cr
#'   Point given as a numeric vector.
#' @param B [\code{matrix}]\cr
#'   Point set (each row corresponds to a point).
#' @template arg_asemoa_dist_fun
#' @return [\code{numeric(1)}]
#' @export
computeDistanceFromPointToSetOfPoints = function(a, B, dist.fun = computeEuclideanDistance) {
  dists = computeDistancesFromPointToSetOfPoints(a, B, dist.fun)
  return(min(dists))
}

computeDistancesFromPointToSetOfPoints = function(a, B, dist.fun = computeEuclideanDistance) {
  # to avoid loops here we construct a matrix and make use of R's vector
  # computation qualities
  tmp = matrix(rep(a, each = ncol(B)), nrow = nrow(B), byrow = TRUE)
  dists = apply(tmp - B, 2L, function(x) {
    dist.fun(x)
  })
  return(dists)
}