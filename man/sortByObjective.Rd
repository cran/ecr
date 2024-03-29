% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pareto.utils.R
\name{sortByObjective}
\alias{sortByObjective}
\alias{sortByObjective.data.frame}
\alias{sortByObjective.matrix}
\alias{sortByObjective.ecr_multi_objective_result}
\alias{sortByObjective.list}
\title{Sort Pareto-front approximation by objective.}
\usage{
sortByObjective(x, obj = 1L, ...)

\method{sortByObjective}{data.frame}(x, obj = 1L, ...)

\method{sortByObjective}{matrix}(x, obj = 1L, ...)

\method{sortByObjective}{ecr_multi_objective_result}(x, obj = 1L, ...)

\method{sortByObjective}{list}(x, obj = 1L, ...)
}
\arguments{
\item{x}{[\code{object}]\cr
Object of type data frame (objectives column-wise), matrix (objectives row-wise),
\code{\link[=ecr_result]{ecr_multi_objective_result}} or \code{list} (with components \dQuote{pareto.front})
and \dQuote{pareto.set}.}

\item{obj}{[\code{integer(1) | character(1)}]\cr
Either the row/column number to sort by or the column name, e.g., for data frames.}

\item{...}{[any]\cr
Further arguments passed down to \code{\link[base]{order}}.}
}
\value{
Modified object.
}
\description{
Sort Pareto-front approximation by objective.
}
