using DataFrames
using Statistics
using Random
using CSV
using Base.Threads

include("../../code/general/preprocessing.jl")
include("../../code/general/utils_general.jl")
include("../../code/general/train_metrics.jl")
include("../../code/general/model_factory.jl")
include("../../code/ann/build_train.jl")
include("../../code/mlj_models/train_mlj.jl")
include("../../code/mlj_models/models.jl")

using LinearAlgebra

function run_approach_experiments(
    approach_name::String,
    model_configs::Dict,
    train_inputs::AbstractArray{<:Real,2},
    train_targets::Vector{String},
    test_inputs::AbstractArray{<:Real,2},
    test_targets::Vector{String};
    k_folds::Int=3, # Reduced to 3 for 100k samples (5 is overkill and slow)
    rng::AbstractRNG=MersenneTwister(1234),
    normalize::Union{Symbol, Nothing}=:zero,
    preprocessing::Union{Nothing, Dict}=nothing
)
    # 1. OPTIMIZATION: Set inner Linear Algebra to single-threaded 
    # to avoid fighting with the outer parallel loop.
    # We will restore this at the end.
    original_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1) 

    results_df = DataFrame(
        Approach = String[], Model = Symbol[], Config = String[],
        Accuracy = Float64[], ErrorRate = Float64[], Sensitivity = Float64[],
        Specificity = Float64[], PPV = Float64[], NPV = Float64[], F1 = Float64[]
    )

    println("=" ^ 80)
    println("APPROACH: $approach_name | Threads: $(Threads.nthreads())")
    println("=" ^ 80)

    # --- Preprocessing (Same as before) ---
    # Note: Explicitly converting to Float32 helps memory speed significantly
    train_inputs_processed = Float32.(train_inputs) 
    test_inputs_processed = Float32.(test_inputs)
    preprocessing_model = nothing

    if normalize
        norm_params = calculateMinMaxNormalizationParameters(train_inputs_processed)
        normalizeMinMax!(train_inputs_processed)
        test_inputs_processed = normalizeMinMax(test_inputs_processed, norm_params)
    elseif normalize == :zero
        norm_params = calculateZeroMeanNormalizationParameters(train_inputs_processed)
        normalizeZeroMean!(train_inputs_processed)
        
        # Normalize test data using training parameters
        test_inputs_processed = normalizeZeroMean(test_inputs_processed, norm_params)
    end
    if preprocessing !== nothing
        println("\n--- Applying preprocessing: $(preprocessing[:type]) ---")
        if preprocessing[:type] == :PCA
            maxoutdim = get(preprocessing, :maxoutdim, nothing)
            variance_ratio = get(preprocessing, :variance_ratio, nothing)
            train_inputs_processed, preprocessing_model = apply_pca_mlj(train_inputs_processed; maxoutdim=maxoutdim, variance_ratio=variance_ratio)
            if get(preprocessing, :apply_to_test, true)
                test_inputs_processed = transform_pca_mlj(preprocessing_model, test_inputs_processed)
            end
        elseif preprocessing[:type] == :LDA
            outdim = get(preprocessing, :outdim, nothing)
            train_inputs_processed, preprocessing_model = apply_lda_mlj(train_inputs_processed, train_targets; outdim=outdim)
            if get(preprocessing, :apply_to_test, true)
                test_inputs_processed = transform_lda_mlj(preprocessing_model, test_inputs_processed)
            end
        end
        println("  Reduced to $(size(train_inputs_processed, 2)) features")
    end

    dataset = (train_inputs_processed, train_targets)
    cv_indices = crossvalidation(train_targets, k_folds, rng)

    # --- Flatten Configs ---
    config_list = []
    for mt in keys(model_configs)
        for cfg in model_configs[mt]
            push!(config_list, (mt, cfg))
        end
    end

    # Lock for thread-safe writing to DataFrame
    df_lock = ReentrantLock()

    # 2. OPTIMIZATION: Use :dynamic scheduling
    # This ensures if one thread gets a "fast" model, it grabs another task immediately
    # rather than waiting for the "slow" threads to finish.
    Threads.@threads :dynamic for i in 1:length(config_list)
        mt, cfg = config_list[i]
        
        # Simple logging to prevent console spam overlap
        # println("Thread $(Threads.threadid()) starting $mt...") 

        # Run CV
        results = modelCrossValidation(mt, cfg, dataset, cv_indices)
        
        # Create row
        local_df = DataFrame(
            Approach = [approach_name],
            Model = [mt],
            Config = [string(cfg)],
            Accuracy = [results[1][1]],
            ErrorRate = [results[2][1]],
            Sensitivity = [results[3][1]],
            Specificity = [results[4][1]],
            PPV = [results[5][1]],
            NPV = [results[6][1]],
            F1 = [results[7][1]]
        )

        # Thread-safe write to main DataFrame
        lock(df_lock) do
            append!(results_df, local_df)
            println("Finished $mt ($cfg) -> Acc: $(round(results[1][1], digits=4))")
        end
    end

    # --- Find Best Configs ---
    best_configs = Dict{Symbol, Any}()
    for sub_df in groupby(results_df, :Model)
        best_row = sub_df[argmax(sub_df.Accuracy), :]
        # We need to parse the config string back or lookup, 
        # but for now let's just print it.
        # Ideally, store the index of the best config to retrieve the object.
        model_type = sub_df.Model[1]
        println("Best $model_type: $(best_row.Config) -> $(round(best_row.Accuracy, digits=4))")
    end

    # Restore BLAS threads
    BLAS.set_num_threads(original_blas_threads)

    return (results_df, best_configs, preprocessing_model)
end



function save_results_to_csv(results_df::DataFrame, filepath::String)
    # 1. Create directory if it doesn't exist (e.g., if you save to "results/my_file.csv")
    dir_path = dirname(filepath)
    if !isempty(dir_path) && !isdir(dir_path)
        mkpath(dir_path)
        println("Created directory: $dir_path")
    end

    # 2. Write the DataFrame to CSV
    CSV.write(filepath, results_df)
    println("✅ Results successfully saved to: $filepath")
end
