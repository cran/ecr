% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/emoa.nsga2.R
\name{nsga2}
\alias{nsga2}
\title{Implementation of the NSGA-II EMOA algorithm by Deb.}
\usage{
nsga2(
  fitness.fun,
  n.objectives = NULL,
  n.dim = NULL,
  minimize = NULL,
  lower = NULL,
  upper = NULL,
  mu = 100L,
  lambda = mu,
  mutator = setup(mutPolynomial, eta = 25, p = 0.2, lower = lower, upper = upper),
  recombinator = setup(recSBX, eta = 15, p = 0.7, lower = lower, upper = upper),
  terminators = list(stopOnIters(100L)),
  ...
)
}
\arguments{
\item{fitness.fun}{[\code{function}]\cr
The fitness function.}

\item{n.objectives}{[\code{integer(1)}]\cr
Number of objectives of \code{obj.fun}.
Optional if \code{obj.fun} is a benchmark function from package \pkg{smoof}.}

\item{n.dim}{[\code{integer(1)}]\cr
Dimension of the decision space.}

\item{minimize}{[\code{logical(n.objectives)}]\cr
Logical vector with ith entry \code{TRUE} if the ith objective of \code{fitness.fun}
shall be minimized. If a single logical is passed, it is assumed to be valid
for each objective.}

\item{lower}{[\code{numeric}]\cr
Vector of minimal values for each parameter of the decision space in case
of float or permutation encoding.
Optional if \code{obj.fun} is a benchmark function from package \pkg{smoof}.}

\item{upper}{[\code{numeric}]\cr
Vector of maximal values for each parameter of the decision space in case
of float or permutation encoding.
Optional if \code{obj.fun} is a benchmark function from package \pkg{smoof}.}

\item{mu}{[\code{integer(1)}]\cr
Number of individuals in the population.
Default is 100.}

\item{lambda}{[\code{integer(1)}]\cr
Offspring size, i.e., number of individuals generated by variation operators
in each iteration.
Default is 100.}

\item{mutator}{[\code{ecr_mutator}]\cr
Mutation operator of type \code{ecr_mutator}.}

\item{recombinator}{[\code{ecr_recombinator}]\cr
Recombination operator of type \code{ecr_recombinator}.}

\item{terminators}{[\code{list}]\cr
List of stopping conditions of type \dQuote{ecr_terminator}.
Default is to stop after 100 iterations.}

\item{...}{[any]\cr
Further arguments passed down to fitness function.}
}
\value{
[\code{ecr_multi_objective_result}]
}
\description{
The NSGA-II merges the current population and the generated offspring and
reduces it by means of the following procedure: It first applies the non
dominated sorting algorithm to obtain the nondominated fronts. Starting with
the first front, it fills the new population until the i-th front does not fit.
It then applies the secondary crowding distance criterion to select the missing
individuals from the i-th front.
}
\note{
This is a pure R implementation of the NSGA-II algorithm. It hides the regular
ecr interface and offers a more R like interface while still being quite
adaptable.
}
\references{
Deb, K., Pratap, A., and Agarwal, S. A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6 (8) (2002),
182-197.
}
\keyword{optimize}
