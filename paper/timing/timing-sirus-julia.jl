using SIRUS, MLJBase;
using CSV, DataFrames, CategoricalArrays;
using StableRNGs: StableRNG;

using Dates
using Random
using Statistics
#using MLJ
#using MLJLinearModels
using Base.Threads: @spawn

using .Sys
#import Sys: total_memory

function benchmark_fit(model, X, y)
    # Prepare machine (MLJ convention)
    mach = machine(model, X, y)

    # Measure time
    start_time = time()

    # Fit the model
    fit!(mach)

    elapsed_time = time() - start_time

    return mach, elapsed_time
end

# Example usage
function demo(X_train, y_train)
    # Synthetic dataset
    #X, y = @load_iris
    model = StableRulesClassifier(;n_trees=1000,partial_sampling=1.0, q=10, max_depth=2,#n_subfeatures=14,
                                     lambda=1, max_rules=25,rng=StableRNG(1));

    fitted_model, fit_time = benchmark_fit(model, X_train, y_train)

    println("Training time: $(round(fit_time, digits=4)) s")
    return fit_time
end

output_dir_data_set = "paper/reproduce-exp/sim-data-time/";
output_dir_Rules = "paper/timing/times-csv/";
X_train_original = CSV.read(joinpath(output_dir_data_set, "X_train.csv"), DataFrame,header=false,delim=",");
y_train_original = CSV.read(joinpath(output_dir_data_set, "y_train.csv"), DataFrame,header=false,delim=",");

fit_times_samples = [];
run_list = [0,1,2,3,4];
for run in run_list
    println("********* Run number SAMPLES: $run *********");
    n_samples_train_list = [100000, 200000, 300000, 400000, 500000];
    n_dim=200;
    #n_samples_train_list = [1000, 2000, 3000, 4000, 5000];
    #n_dim=10;
    for n_samples_train in n_samples_train_list
        X_train = X_train_original[1:n_samples_train, names(X_train_original)[1:n_dim]];
        y_train_df = y_train_original[1:n_samples_train,r"Column1"];
        y_train_df = coerce(y_train_df, :Column1 => Multiclass);
        y_train_vector = y_train_df.Column1;

        println("Benchmark n_samples: $(round(n_samples_train, digits=4)) samples, n_dim: $(round(n_dim, digits=4)) features");
        fit_time = demo(X_train, y_train_vector);
        push!(fit_times_samples, fit_time);
    end
end

# Save fit_times_samples to CSV
CSV.write(joinpath(output_dir_Rules, "list_time_samples_julia.csv"), Tables.table(fit_times_samples); writeheader=false)

fit_times_dims = [];
run_list = [0,1,2,3,4];
for run in run_list
    println("********* Run number DIMENSION: $run *********");
    n_samples_train = 300000;
    n_dims = [15, 25, 50, 75, 100, 125, 150, 175, 200];
    #n_samples_train = 3000;
    #n_dims = [5, 6, 7, 8, 9, 10, 11, 12, 13];
    for n_dim in n_dims
        X_train = X_train_original[1:n_samples_train, names(X_train_original)[1:n_dim]];
        y_train_df = y_train_original[1:n_samples_train,r"Column1"];
        y_train_df = coerce(y_train_df, :Column1 => Multiclass);
        y_train_vector = y_train_df.Column1;

        println("Benchmark n_samples: $(round(n_samples_train, digits=4)) samples, n_dim: $(round(n_dim, digits=4)) features");
        fit_time = demo(X_train, y_train_vector);
        push!(fit_times_dims, fit_time);
    end
end

CSV.write(joinpath(output_dir_Rules, "list_time_dims_julia.csv"), Tables.table(fit_times_dims); writeheader=false)
