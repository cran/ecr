% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PA.EMOA.plotScatter2d.R
\name{plotScatter2d}
\alias{plotScatter2d}
\title{Visualize bi-objective Pareto-front approximations.}
\usage{
plotScatter2d(
  df,
  obj.cols = c("f1", "f2"),
  shape = "algorithm",
  colour = NULL,
  highlight.algos = NULL,
  offset.highlighted = 0,
  title = NULL,
  subtitle = NULL,
  facet.type = "wrap",
  facet.args = list()
)
}
\arguments{
\item{df}{[\code{data.frame}]\cr
Data.frame with columns at least \code{obj.cols}, \dQuote{prob} and \dQuote{algorithm}.}

\item{obj.cols}{[\code{character(>= 2)}]\cr
Column names of the objective functions.
Default is \code{c("f1", "f2")}.}

\item{shape}{[\code{character(1)}]\cr
Name of column which shall be used to define shape of points.
Default is \dQuote{algorithm}.}

\item{colour}{[\code{character(1)}]\cr
Name of column which shall be used to define colour of points.
Default is \code{NULL}, i.e., coloring is deactivated.}

\item{highlight.algos}{[\code{character(1)}]\cr
Name of algorithm to highlight exclusively. Useful to highlight, e.g., the
true Pareto-optimal front (if known) or some reference set.
Default is \code{NULL}, i.e., unknown.}

\item{offset.highlighted}{[\code{numeric(1)}]\cr
Numeric offset used to shift set (see \code{highlight.algos})
which should be highlighted.
Even though this produces objective vectors it
may be used to make visible reference sets which otherwise would
be hidden by overlap of multiple other approximation sets.}

\item{title}{[\code{character(1)}]\cr
Plot title.}

\item{subtitle}{[\code{character(1)}]\cr
Plot subtitle.}

\item{facet.type}{[\code{character(1)}]\cr
Which faceting method to use? Pass \dQuote{wrap} for \code{\link[ggplot2]{facet_wrap}}
or \dQuote{grid} for \code{\link[ggplot2]{facet_grid}}.
Default is \dQuote{wrap}.}

\item{facet.args}{[\code{list}]\cr
Named list of arguments passed down to \code{\link[ggplot2]{facet_wrap}} or
\code{\link[ggplot2]{facet_grid}} respectively (depends on \code{facet.type}).
E.g., \code{nrow} to change layout.
Default is the empty list. In this case data is grouped by problem.}
}
\value{
[\code{\link[ggplot2]{ggplot}}] A ggplot object.
}
\description{
Given a data frame with the results of (multiple) runs of (multiple)
different multi-objective optimization algorithms on (multiple) problem instances
the function generates \code{\link[ggplot2]{ggplot}} plots of the obtained
Pareto-front approximations.
}
\note{
At the moment only approximations of bi-objective functions are supported.
}
\examples{
\dontrun{
# load examplary data
data(mcMST)
print(head(mcMST))

# no customization; use the defaults
pl = plotFronts(mcMST)

# algo PRIM is obtained by weighted sum scalarization
# Since the front is (mainly) convex we highlight these solutions
pl = plotFronts(mcMST, highlight.algos = "PRIM")

# customize layout
pl = plotFronts(mcMST, title = "Pareto-approximations",
  subtitle = "based on different mcMST algorithms.", facet.args = list(nrow = 2))
}
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
\code{\link{plotFront}()},
\code{\link{plotScatter3d}()},
\code{\link{toLatex}()}
}
\concept{EMOA performance assessment tools}
