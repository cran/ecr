% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/initPopulation.R
\name{initPopulation}
\alias{initPopulation}
\title{Helper function to build initial population.}
\usage{
initPopulation(mu, gen.fun, initial.solutions = NULL, ...)
}
\arguments{
\item{mu}{[\code{integer(1)}]\cr
Number of individuals in the population.}

\item{gen.fun}{[\code{function}]\cr
Function used to generate initial solutions, e.g., \code{\link{genBin}}.}

\item{initial.solutions}{[\code{list}]\cr
List of individuals which should be placed in the initial population.
If the number of passed individuals
is lower than \code{mu}, the population will be filled up
by individuals generated by the corresponding generator.
Default is \code{NULL}, i.e., the entire population is generated by the
population generator.}

\item{...}{[any]\cr
Further parameters passed to \code{gen.fun}.}
}
\value{
[\code{ecr_population}]
}
\description{
Generates the initial population. Optionally a set of initial solutions
can be passed.
}
