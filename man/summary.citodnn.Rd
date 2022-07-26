% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dnn.R
\name{summary.citodnn}
\alias{summary.citodnn}
\title{Summarize Neural Network of class citodnn}
\usage{
\method{summary}{citodnn}(object, n_permute = 256, ...)
}
\arguments{
\item{object}{a model of class citodnn created by \code{\link{dnn}}}

\item{n_permute}{number of permutations performed, higher equals more accurate importance results}

\item{...}{additional arguments}
}
\value{
summary.glm returns an object of class "summary.citodnn", a list with components
}
\description{
Performs a Feature Importance calculation based on Permutations
}
\details{
Performs the feature importance calculation as suggested by  Fisher, Rudin, and Dominici (2018).
For each feature n permutation get done and original and permuted predictive mean squared error (\eqn{e_{perm}} & \eqn{e_{orig}}) get evaluated with \eqn{ FI_j= e_{perm}/e_{orig}}. Based on Mean Squared Error.
}
