% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/paretoArchive.R
\name{initParetoArchive}
\alias{initParetoArchive}
\title{Initialize Pareto Archive.}
\usage{
initParetoArchive(control, max.size = Inf, trunc.fun = NULL)
}
\arguments{
\item{control}{[\code{ecr_control}]\cr
Control object.}

\item{max.size}{[\code{integer(1)}]\cr
Maximum capacity of the Pareto archive, i.e., the maximal number of non-dominated
points which can be stored in the archive. Default is \code{Inf}, i.e., (theoretically)
unbounded capacity.}

\item{trunc.fun}{[\code{function(archive, inds, fitness, ...)}]\cr
In case the archive is limited in capacity, i.e., \code{max.size} is not infinite,
this function is called internally if an archive overflow occurs. This function
expects the \code{archive}, a list of individuals \code{inds}, a matrix of fitness
values (each column contains the fitness value(s) of one individual) \code{fitness}
and further optional arguments \code{...} which may be used by the internals
of \code{trunc.fun}. The function must return a list with components \dQuote{fitness}
and \dQuote{inds} which shall be the subsets of \code{fitness} and \code{inds}
respectively, which should be kept by the archive.}
}
\value{
[\code{ecr_pareto_archive}]
}
\description{
A Pareto archive is usually used to store all / a part of the
non-dominated points stored during a run of an multi-objective evolutionary
algorithm.
}
\seealso{
Other ParetoArchive: 
\code{\link{getIndividuals}()},
\code{\link{getSize}()},
\code{\link{updateParetoArchive}()}
}
\concept{ParetoArchive}
