using DataFrames
using Statistics
using Random

include("../../code/general/preprocessing.jl")
include("../../code/general/utils_general.jl")
include("../../code/general/train_metrics.jl")
include("../../code/general/model_factory.jl")
include("../../code/ann/build_train.jl")
include("../../code/mlj_models/train_mlj.jl")
include("../../code/mlj_models/models.jl")

"""
    run_approach_experiments(approach_name, model_configs, train_inputs, train_targets, 
                            test_inputs, test_targets; kwargs...)

Run experiments for a given approach with configurable models and preprocessing.

# Arguments
- `approach_name::String`: Name of the approach (e.g., "Full Features", "PCA Reduced")
- `model_configs::Dict`: Dictionary mapping model types to their configurations
  - Keys: `:ANN`, `:SVM`, `:DT`, `:KNN`, `:RF`, `:AdaBoost`, `:CatBoost`
  - Values: Arrays of configuration dictionaries
- `train_inputs::Matrix{Float64}`: Training input features
- `train_targets::Vector{String}`: Training target labels
- `test_inputs::Matrix{Float64}`: Test input features  
- `test_targets::Vector{String}`: Test target labels

# Keyword Arguments
- `k_folds::Int=5`: Number of folds for cross-validation
- `rng::AbstractRNG`: Random number generator
- `normalize::Bool=true`: Whether to normalize features
- `preprocessing::Union{Nothing, Dict}=nothing`: Preprocessing configuration
  - `:type => :PCA` or `:LDA`
  - `:n_components => Int`: Number of components to keep
  - `:apply_to_test::Bool=true`: Whether to apply preprocessing to test set

# Returns
- `results_df::DataFrame`: Results for all model configurations
- `best_configs::Dict`: Best configuration for each model type
"""


function run_approach_experiments(
    approach_name::String,
    model_configs::Dict,
    train_inputs::AbstractArray{<:Real,2},
    train_targets::Vector{String},
    test_inputs::AbstractArray{<:Real,2},
    test_targets::Vector{String};
    k_folds::Int=5,
    rng::AbstractRNG=MersenneTwister(1234),
    normalize::Bool=true,
    preprocessing::Union{Nothing, Dict}=nothing
)
    results_df = DataFrame(
        Approach = String[],
        Model = Symbol[],
        Config = String[],
        Accuracy = Float64[],
        F1 = Float64[],
        Sensitivity = Float64[],
        Specificity = Float64[]
    )

    println("=" ^ 80)
    println("APPROACH: $approach_name")
    println("=" ^ 80)
    
    # Apply preprocessing if specified
    train_inputs_processed = copy(train_inputs)
    test_inputs_processed = copy(test_inputs)
    preprocessing_model = nothing
    
    if preprocessing !== nothing
        println("\n--- Applying preprocessing: $(preprocessing[:type]) ---")
        if preprocessing[:type] == :PCA
            n_components = get(preprocessing, :n_components, nothing)
            variance_ratio = get(preprocessing, :variance_ratio, nothing)
            train_inputs_processed, preprocessing_mach = apply_pca_mlj(
                train_inputs; 
                n_components=n_components, 
                variance_ratio=variance_ratio
            )
            if get(preprocessing, :apply_to_test, true)
                test_inputs_processed = transform_pca_mlj(preprocessing_mach, test_inputs)
            end
            println("  Reduced from $(size(train_inputs, 2)) to $(size(train_inputs_processed, 2)) features")
        elseif preprocessing[:type] == :LDA
            n_components = get(preprocessing, :outdim, nothing)
            train_inputs_processed, preprocessing_mach = apply_lda_mlj(
                train_inputs,
                train_targets;
                outdim=outdim
            )
            if get(preprocessing, :apply_to_test, true)
                test_inputs_processed = transform_lda_mlj(preprocessing_mach, test_inputs)
            end
            println("  Reduced from $(size(train_inputs, 2)) to $(size(train_inputs_processed, 2)) features")
        end
    end
    
    # Normalize training data
    if normalize
        normalizeMinMax!(train_inputs_processed)
        norm_params = calculateMinMaxNormalizationParameters(train_inputs_processed)
        
        # Normalize test data using training parameters
        test_inputs_processed = normalizeMinMax(test_inputs_processed, norm_params)
    end
    
    # Prepare dataset tuple
    train_inputs_f32 = Float32.(train_inputs_processed)
    dataset = (train_inputs_f32, train_targets)
    
    # Generate cross-validation indices
    cv_indices = crossvalidation(train_targets, k_folds, rng)
    
    # ========================================================================
    # HYPERPARAMETER TESTING
    # ========================================================================
    
    println("\n--- Testing different model configurations ---")
    
    # Store best results
    best_configs = Dict{Symbol, Any}()
    best_accs = Dict{Symbol, Float64}()
    
    # Initialize best accuracies
    for model_type in keys(model_configs)
        best_accs[model_type] = -1.0
        best_configs[model_type] = nothing
    end
    
    # Test ANN
    if haskey(model_configs, :ANN) && !isempty(model_configs[:ANN])
        println("\nTesting ANN configurations...")
        for (i, config) in enumerate(model_configs[:ANN])
            println("  ANN Config $i/$(length(model_configs[:ANN])): topology=$(config[:topology])")
            results = modelCrossValidation(:ANN, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:ANN]
                best_accs[:ANN] = acc_mean
                best_configs[:ANN] = config
            end
            push!(results_df, (
                approach_name,
                :ANN,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test SVM
    if haskey(model_configs, :SVM) && !isempty(model_configs[:SVM])
        println("\nTesting SVM configurations...")
        for (i, config) in enumerate(model_configs[:SVM])
            println("  SVM Config $i/$(length(model_configs[:SVM])): kernel=$(config[:kernel]), cost=$(config[:cost])")
            results = modelCrossValidation(:SVC, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:SVM]
                best_accs[:SVM] = acc_mean
                best_configs[:SVM] = config
            end
            push!(results_df, (
                approach_name,
                :SVM,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test Decision Tree
    if haskey(model_configs, :DT) && !isempty(model_configs[:DT])
        println("\nTesting Decision Tree configurations...")
        for (i, config) in enumerate(model_configs[:DT])
            println("  DT Config $i/$(length(model_configs[:DT])): max_depth=$(config[:max_depth])")
            results = modelCrossValidation(:DecisionTreeClassifier, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:DT]
                best_accs[:DT] = acc_mean
                best_configs[:DT] = config
            end
            push!(results_df, (
                approach_name,
                :DT,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test kNN
    if haskey(model_configs, :KNN) && !isempty(model_configs[:KNN])
        println("\nTesting kNN configurations...")
        for (i, config) in enumerate(model_configs[:KNN])
            println("  kNN Config $i/$(length(model_configs[:KNN])): k=$(config[:n_neighbors])")
            results = modelCrossValidation(:KNeighborsClassifier, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:KNN]
                best_accs[:KNN] = acc_mean
                best_configs[:KNN] = config
            end
            push!(results_df, (
                approach_name,
                :KNN,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test Random Forest
    if haskey(model_configs, :RF) && !isempty(model_configs[:RF])
        println("\nTesting Random Forest configurations...")
        for (i, config) in enumerate(model_configs[:RF])
            println("  RF Config $i/$(length(model_configs[:RF])): n_trees=$(get(config, :n_trees, 100))")
            results = modelCrossValidation(:RandomForestClassifier, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:RF]
                best_accs[:RF] = acc_mean
                best_configs[:RF] = config
            end
            push!(results_df, (
                approach_name,
                :RF,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test AdaBoost
    if haskey(model_configs, :AdaBoost) && !isempty(model_configs[:AdaBoost])
        println("\nTesting AdaBoost configurations...")
        for (i, config) in enumerate(model_configs[:AdaBoost])
            println("  AdaBoost Config $i/$(length(model_configs[:AdaBoost])): n_estimators=$(get(config, :n_estimators, 50))")
            results = modelCrossValidation(:AdaBoostClassifier, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:AdaBoost]
                best_accs[:AdaBoost] = acc_mean
                best_configs[:AdaBoost] = config
            end
            push!(results_df, (
                approach_name,
                :AdaBoost,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    # Test CatBoost
    if haskey(model_configs, :CatBoost) && !isempty(model_configs[:CatBoost])
        println("\nTesting CatBoost configurations...")
        for (i, config) in enumerate(model_configs[:CatBoost])
            println("  CatBoost Config $i/$(length(model_configs[:CatBoost])): iterations=$(get(config, :iterations, 100))")
            results = modelCrossValidation(:CatBoostClassifier, config, dataset, cv_indices)
            acc_mean = results[1][1]
            println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
            if acc_mean > best_accs[:CatBoost]
                best_accs[:CatBoost] = acc_mean
                best_configs[:CatBoost] = config
            end
            push!(results_df, (
                approach_name,
                :CatBoost,
                string(config),
                results[1][1],
                results[7][1],
                results[3][1],
                results[4][1]
            ))
        end
    end
    
    println("\n--- Best configurations found ---")
    for (model_type, config) in best_configs
        if config !== nothing
            acc = best_accs[model_type]
            println("  Best $model_type: $(string(config)) - Accuracy: $(round(acc, digits=4))")
        end
    end
    
    # ========================================================================
    # ENSEMBLE TRAINING
    # ========================================================================
    
    println("\n--- Training ensemble model (combining best models) ---")
    
    # Select best models for ensemble (at least 2, up to 3)
    println("\n--- Training ensemble model (combining best models) ---")
    
    # Select best models for ensemble - only DTs, SVMs, and KNNs
    available_models = [:SVM, :DT, :KNN]
    ensemble_candidates = [(model_type, best_configs[model_type], best_accs[model_type]) 
                          for model_type in available_models 
                          if haskey(best_configs, model_type) && best_configs[model_type] !== nothing]
    
    # Sort by accuracy and take all available (up to 3)
    sort!(ensemble_candidates, by=x -> x[3], rev=true)
    num_ensemble = length(ensemble_candidates)
    
    if num_ensemble >= 2
        ensemble_estimators = [Symbol(string(c[1]) * "Classifier") for c in ensemble_candidates[1:num_ensemble]]
        # Map model types to their MLJ symbol names
        model_type_map = Dict(
            :SVM => :SVC,
            :DT => :DecisionTreeClassifier,
            :KNN => :KNeighborsClassifier,
            :RF => :RandomForestClassifier,
            :AdaBoost => :AdaBoostClassifier,
            :CatBoost => :CatBoostClassifier
        )
        ensemble_estimators = [model_type_map[c[1]] for c in ensemble_candidates[1:num_ensemble]]
        
        # Prepare training data for ensemble (needs Bool array)
        classes = unique(train_targets)
        train_targets_bool = oneHotEncoding(train_targets, classes)
        
        # Build hyperparameters dict
        ensemble_hyperparams = Dict()
        for (model_type, config, _) in ensemble_candidates[1:num_ensemble]
            mlj_symbol = model_type_map[model_type]
            ensemble_hyperparams[mlj_symbol] = config
        end
        
        ensemble_config = Dict(:rng => rng)
        
        println("Training Stack ensemble with $num_ensemble models: $(ensemble_estimators)...")
        ensemble_results = trainClassEnsemble(
            ensemble_estimators,
            ensemble_hyperparams,
            ensemble_config,
            (train_inputs_processed, train_targets_bool),
            cv_indices
        )
        
        println("Ensemble Results:")
        println("  Accuracy: $(round(ensemble_results[1][1], digits=4)) ± $(round(ensemble_results[1][2], digits=4))")
        println("  F1-Score: $(round(ensemble_results[7][1], digits=4)) ± $(round(ensemble_results[7][2], digits=4))")
        push!(results_df, (
            approach_name,
            :Stack,
            string(Dict(:ensemble => ensemble_config, :models => ensemble_hyperparams)),
            ensemble_results[1][1],
            ensemble_results[7][1],
            ensemble_results[3][1],
            ensemble_results[4][1]
        ))
    else
        println("  Not enough models for ensemble (need at least 2)")
    end
    
    return (results_df, best_configs, preprocessing_model)
end
