% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/logger.default.R
\name{initLogger}
\alias{initLogger}
\title{Initialize a log object.}
\usage{
initLogger(
  control,
  log.stats = list(fitness = list("min", "mean", "max")),
  log.extras = NULL,
  log.pop = FALSE,
  init.size = 1000L
)
}
\arguments{
\item{control}{[\code{ecr_control}]\cr
Control object.}

\item{log.stats}{[\code{list}]\cr
List of lists for statistic computation on attributes of the individuals
of the population. Each entry should be named by the attribute it should be
based on, e.g., fitness, and should contain a list of R functions as a
character string or a a list with elements \code{fun} for the function, and \code{pars} for additional
parameters which shall be passed to the corresponding function.
Each function is required to return a scalar numeric value.
By default the minimum, mean and maximum of the fitness values is computed.
Since fitness statistics are the most important ones these do not have to
be stored as attributes, but can be passed as a matrix to the update function.}

\item{log.extras}{[\code{character}]\cr
Possibility to instruct the logger to store additional
scalar values in each generation. Named character vector where the names
indicate the value to store and the value indicates the corresponding data types.
Currently we support all atomic modes of \code{\link[base]{vector}} expect \dQuote{factor}
and \dQuote{raw}.}

\item{log.pop}{[\code{logical(1)}]\cr
Shall the entire population be saved in each generation?
Default is \code{FALSE}.}

\item{init.size}{[\code{integer(1)}]\cr
Initial number of rows of the slot of the logger, where the fitness
statistics are stored. The size of the statistics log is doubled each time an
overflow occurs.
Default is 1000.}
}
\value{
[\code{ecr_logger}]
  An S3 object of class \code{ecr_logger} with the following components:
  \describe{
    \item{log.stats}{The \code{log.stats} list.}
    \item{log.pop}{The \code{log.pop} parameter.}
    \item{init.size}{Initial size of the log.}
    \item{env}{The actual log. This is an R environment which ensures, that
    in-place modification is possible.}
  }
}
\description{
Logging is a central aspect of each EA. Besides the final solution(s)
especially in research often we need to keep track of different aspects of the
evolutionary process, e.g., fitness statistics. The logger of ecr keeps
track of different user-defined statistics and the population.
It may also be used to check stopping conditions (see \code{makeECRTerminator}). Most
important this logger is used internally by the \code{\link{ecr}} black-box interface.
}
\note{
Statistics are logged in a \code{data.frame}.
}
\examples{
control = initECRControl(function(x) sum(x), minimize = TRUE,
  n.objectives = 1L)
control = registerECROperator(control, "mutate", mutBitflip, p = 0.1)
control = registerECROperator(control, "selectForMating", selTournament, k = 2)
control = registerECROperator(control, "selectForSurvival", selGreedy)

log = initLogger(control,
  log.stats = list(
    fitness = list("mean", "myRange" = function(x) max(x) - min(x)),
    age = list("min", "max")
  ), log.pop = TRUE, init.size = 1000L)

 # simply pass stuff down to control object constructor
population = initPopulation(mu = 10L, genBin, n.dim = 10L)
fitness = evaluateFitness(control, population)

# append fitness to individuals and init age
for (i in seq_along(population)) {
  attr(population[[i]], "fitness") = fitness[, i]
  attr(population[[i]], "age") = 1L
}

for (iter in seq_len(10)) {
  # generate offspring
  offspring = generateOffspring(control, population, fitness, lambda = 5)
  fitness.offspring = evaluateFitness(control, offspring)

  # update age of population
  for (i in seq_along(population)) {
    attr(population[[i]], "age") = attr(population[[i]], "age") + 1L
  }

  # set offspring attributes
  for (i in seq_along(offspring)) {
    attr(offspring[[i]], "fitness") = fitness.offspring[, i]
    # update age
    attr(offspring[[i]], "age") = 1L
  }

  sel = replaceMuPlusLambda(control, population, offspring)

  population = sel$population
  fitness = sel$fitness

  # do some logging
  updateLogger(log, population, n.evals = 5)
}
head(getStatistics(log))
}
\seealso{
Other logging: 
\code{\link{getPopulationFitness}()},
\code{\link{getPopulations}()},
\code{\link{getStatistics}()},
\code{\link{updateLogger}()}
}
\concept{logging}
