% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PA.EMOA.plotScatter.R
\name{plotFront}
\alias{plotFront}
\title{Draw scatterplot of Pareto-front approximation}
\usage{
plotFront(x, obj.names = NULL, minimize = TRUE, ...)
}
\arguments{
\item{x}{[\code{matrix} | \code{data.frame}]\cr
Object which contains the approximations sets.}

\item{obj.names}{[\code{character}]\cr
Optional objectives names.
Default is \code{c("f1", "f2")}.}

\item{minimize}{[\code{logical}]\cr
Logical vector with ith entry \code{TRUE} if the ith objective shall be minimized.
If a single logical is passed, it is assumed to be valid for each objective.
If the matrix is of type \code{ecr_fitness_matrix} (this is the case if it is
produced by one of ecr2's utility functions, e.g. \code{\link{evaluateFitness}}),
the appended \code{minimize} attribute is the default.}

\item{...}{[any]\cr
Not used at the moment.}
}
\value{
[\code{ggplot}] \pkg{ggplot} object.
}
\description{
The function expects a data.frame or a matrix. By default the first
2 or 3 columns/rows are assumed to contain the elements of the approximation sets.
Depending on the number of numeric columns (in case of a data.frame) or the
number of rows (in case of a matrix) the function internally calls
\code{\link{plotScatter2d}} or \code{plotScatter3d}.
}
\seealso{
Other EMOA performance assessment tools: 
\code{\link{approximateNadirPoint}()},
\code{\link{approximateRefPoints}()},
\code{\link{approximateRefSets}()},
\code{\link{computeDominanceRanking}()},
\code{\link{emoaIndEps}()},
\code{\link{makeEMOAIndicator}()},
\code{\link{niceCellFormater}()},
\code{\link{normalize}()},
\code{\link{plotDistribution}()},
\code{\link{plotScatter2d}()},
\code{\link{plotScatter3d}()},
\code{\link{toLatex}()}
}
\concept{EMOA performance assessment tools}
