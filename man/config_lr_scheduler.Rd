% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/training.R
\name{config_lr_scheduler}
\alias{config_lr_scheduler}
\title{Creation of customized learning rate scheduler objects}
\usage{
config_lr_scheduler(
  type = c("lambda", "multiplicative", "reduce_on_plateau", "one_cycle", "step"),
  verbose = FALSE,
  ...
)
}
\arguments{
\item{type}{String defining which type of scheduler should be used. See Details.}

\item{verbose}{If TRUE, additional information about scheduler will be printed to console.}

\item{...}{additional arguments to be passed to scheduler. See Details.}
}
\value{
object of class cito_lr_scheduler to give to \code{\link{dnn}}
}
\description{
Helps create custom learning rate schedulers for \code{\link{dnn}}.
}
\details{
different learning rate scheduler need different variables, these functions will tell you which variables can be set:
\itemize{
\item lambda: \code{\link[torch]{lr_lambda}}
\item multiplicative: \code{\link[torch]{lr_multiplicative}}
\item reduce_on_plateau: \code{\link[torch]{lr_reduce_on_plateau}}
\item one_cycle: \code{\link[torch]{lr_one_cycle}}
\item step: \code{\link[torch]{lr_step}}
}
}
\examples{
\donttest{
if(torch::torch_is_installed()){
library(cito)

# create learning rate scheduler object
scheduler <- config_lr_scheduler(type = "step",
                        step_size = 30,
                        gamma = 0.15,
                        verbose = TRUE)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris, lr_scheduler = scheduler)

}
}
}
