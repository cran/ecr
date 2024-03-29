% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dominates.R
\name{setDominates}
\alias{setDominates}
\title{Check if one set is better than another.}
\usage{
setDominates(x, y)
}
\arguments{
\item{x}{[\code{matrix}]\cr
First set of points.}

\item{y}{[\code{matrix}]\cr
Second set of points.}
}
\value{
[\code{logical(1)}]
}
\description{
The function checks, whether each points of the second set of points
is dominated by at least one point from the first set.
}
