
# Reprodcing time experiments

This folder enables you to reproduce the experiments from the paper that compare the computation time of three SIRUS implementations: R, Julia, and Python (ours).

You will need to set up a separate environment for each implementation. The instructions below explain how to create these environments and install the required packages.

Note that the provided scripts report only the *total* runtime (forest training + rule extraction). The runtimes of the two intermediate steps—(i) growing the forest and (ii) extracting the rules—are **not** measured dynamically by the scripts; instead, they are hard-coded. This is because, in the R implementation, we modified the source code (see `R/Sirus.R`, starting at line 172) to print both the forest-growing time and the rule-extraction time, installed this patched version in our R environment, then saved the console output to a `.txt` file. We subsequently parsed these logs and hard-coded the resulting values into the `.csv` files located in the `time-csv` folder. We followed the same procedure for the Julia implementation.

To help you benchmark results on your own machine, we provide a configuration that recomputes **only the total fitting time** and updates the corresponding `.csv` files accordingly. The per-step timings (forest growth vs. rule extraction) are included in the paper mainly for pedagogical purposes; the primary metric of interest is the total runtime. If you modify the R or Julia source code to re-enable per-step timing on your setup, you can regenerate those values and update the relevant `.csv` files to plot your own forest-growing and rule-extraction times.


## Fitting time setup
You need to run "time-sirus-python.py" before running "time-sirus-r.R" and time-sirus-julia.jl. Below are the intrusction to follow to succesfully lauch the code.

### Python
Here are the commands for the python script:

```bash
python paper/timing/timing-sirus-python.py
```

### R
You need a specific R environement with sirus installed for the r script. Here a R code for downlading sirus:
```R
to_download_sirus <- TRUE # Set to TRUE if you want to download and install sirus package from GitLab
is_using_colab <- FALSE  # set to TRUE when running in Colab
if (to_download_sirus) {
  # Download the tar.gz
  utils::download.file(
    "https://gitlab.com/drti/sirus/-/archive/sirus_0.3.3/sirus-sirus_0.3.3.tar.gz?ref_type=tags",
    destfile = "sirus-sirus_0.3.3.tar.gz",
    mode = "wb"
  )

  # Rename/move the file
  file.rename(
    from = "sirus-sirus_0.3.3.tar.gz",
    to   = "sirus_0.3.3.tar.gz"
  )
  install.packages("ROCR",repos="http://cran.us.r-project.org")
  install.packages(c("RcppEigen", "glmnet"), repos = "https://cloud.r-project.org")
  install.packages("sirus_0.3.3.tar.gz", repos = NULL)
}
```

Then run:
```bash
Rscript paper/timing/timing-sirus-r.R
```

### Julia
You need a specific julia environement with sirus installed for the julia script. Here a julia code for downlading sirus:
```Julia
using Pkg;
Pkg.add(name="SIRUS",version="2.0.0");
```
Then run in julia:
```Julia
Pkg.add(name="MLJBase",version="0.21")
Pkg.add(name="MLJLinearModels",version="0.10")
Pkg.add(name="MLJModelInterface",version="1.4")
Pkg.add("StableRNGs")
Pkg.add("CategoricalArrays")

Pkg.add("Dates")
Pkg.add("Random")
Pkg.add("Statistics")

Pkg.add("CSV")
Pkg.add("DataFrames")
```

Finally run:

```bash
julia --threads 5 paper/timing/timing-sirus-julia.jl
```
