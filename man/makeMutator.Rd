% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/makeMutator.R
\name{makeMutator}
\alias{makeMutator}
\title{Construct a mutation operator.}
\usage{
makeMutator(mutator, supported = getAvailableRepresentations())
}
\arguments{
\item{mutator}{[\code{function}]\cr
Actual mutation operator.}

\item{supported}{[\code{character}]\cr
Vector of strings/names of supported parameter representations. Possible choices:
\dQuote{permutation}, \dQuote{float}, \dQuote{binary} or \dQuote{custom}.}
}
\value{
[\code{ecr_mutator}]
  Mutator object.
}
\description{
Helper function which constructs a mutator, i. e., a mutation operator.
}
