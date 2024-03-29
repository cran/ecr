% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/makeRecombinator.R
\name{makeRecombinator}
\alias{makeRecombinator}
\title{Construct a recombination operator.}
\usage{
makeRecombinator(
  recombinator,
  supported = getAvailableRepresentations(),
  n.parents = 2L,
  n.children = NULL
)
}
\arguments{
\item{recombinator}{[\code{function}]\cr
Actual mutation operator.}

\item{supported}{[\code{character}]\cr
Vector of strings/names of supported parameter representations. Possible choices:
\dQuote{permutation}, \dQuote{float}, \dQuote{binary} or \dQuote{custom}.}

\item{n.parents}{[\code{integer(1)}]\cr
Number of parents supported.}

\item{n.children}{[\code{integer(1)}]\cr
How many children does the recombinator produce?
Default is \code{1}.}
}
\value{
[\code{ecr_recombinator}]
  Recombinator object.
}
\description{
Helper function which constructs a recombinator, i. e., a recombination operator.
}
\note{
If a recombinator returns more than one child, the \code{multiple.children}
parameter needs to be \code{TRUE}, which is the default. In case of multiple
children produced these have to be placed within a list.
}
