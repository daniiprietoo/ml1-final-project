using Random
using Statistics
using MLJ
using Flux

# Set random seed for reproducibility
const RANDOM_SEED = 1234
Random.seed!(RANDOM_SEED)
const rng = MersenneTwister(RANDOM_SEED)

# Include all necessary modules
include("code/general/preprocessing.jl")
include("code/general/utils_general.jl")
include("code/general/train_metrics.jl")
include("code/general/model_factory.jl")
include("code/ann/build_train.jl")
include("code/mlj_models/train_mlj.jl")
include("code/mlj_models/models.jl")


println("=" ^ 80)
println("Random seed set to: $RANDOM_SEED for reproducibility")
println()

# DATA LOADING AND PREPROCESSING

println("Step 1: Loading and preprocessing data...")
const TRACKS_FILE = "data/tracks.csv"
const FEATURES_FILE = "data/features.csv"

# Load and merge data
df = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens], selected_features_algorithms=[:mfcc])[1:10000, :]
println("Loaded $(nrow(df)) tracks with $(ncol(df) - 2) features")  # -2 for track_id and listens

listens = df.track_listens

# Low (0-33), Medium (34-66), High (67-100)
p33 = quantile(listens, 0.3333)
p66 = quantile(listens, 0.6667)

function create_popularity_class(listen_count)
    if listen_count <= p33
        return "Low"
    elseif listen_count <= p66
        return "Medium"
    else
        return "High"
    end
end

targets = [create_popularity_class(listen) for listen in listens]
println("Popularity:")
println("  Low: $(sum(targets .== "Low"))")
println("  Medium: $(sum(targets .== "Medium"))")
println("  High: $(sum(targets .== "High"))")
println()

# Extract feature columns (exclude track_id and listens)
feature_cols = [col for col in names(df) if col != :track_id && col != :track_listens]
features_df = select(df, feature_cols)

# Check for and handle non-numeric columns
# Convert DataFrame to matrix, handling any string columns
numeric_cols = String[]
for col in names(features_df)
    col_type = eltype(features_df[!, col])
    if col_type <: Number
        push!(numeric_cols, string(col))
    else
        println("Warning: Column $col has type $col_type and will be excluded")
    end
end

# Select only numeric columns and convert to Float64 matrix
if length(numeric_cols) < ncol(features_df)
    features_df = select(features_df, Symbol.(numeric_cols))
end

inputs = Matrix{Float64}(features_df)

println("Input features shape: $(size(inputs))")
println()

# ============================================================================
# TRAIN/TEST SPLIT (Hold-out)
# ============================================================================

println("Step 2: Performing train/test split...")
const TEST_RATIO = 0.2
(trainIndexes, testIndexes) = holdOut(size(inputs, 1), TEST_RATIO, rng)

train_inputs = inputs[trainIndexes, :]
train_targets = targets[trainIndexes]
test_inputs = inputs[testIndexes, :]
test_targets = targets[testIndexes]

println("Training set: $(size(train_inputs, 1)) samples")
println("Test set: $(size(test_inputs, 1)) samples")
println()

# ============================================================================
# HELPER FUNCTION: Run experiments for an approach
# ============================================================================

# ANN Configurations (at least 8 different architectures, 1-2 hidden layers)
ann_configs = [
    Dict(:topology => [10], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3)
    #=
    Dict(:topology => [20], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [30], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [50], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [10, 5], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [20, 10], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [30, 15], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [50, 25], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    =#
]

# SVM Configurations (at least 8 different configurations: kernels + C values)
svm_configs = [
    
    Dict(:kernel => "linear", :cost => 0.1)
    #=
    Dict(:kernel => "linear", :cost => 1.0),
    Dict(:kernel => "rbf", :cost => 0.1, :gamma => 0.01),
    Dict(:kernel => "rbf", :cost => 1.0, :gamma => 0.01),
    Dict(:kernel => "rbf", :cost => 10.0, :gamma => 0.1),
    Dict(:kernel => "poly", :cost => 1.0, :gamma => 0.01, :degree => 2),
    Dict(:kernel => "poly", :cost => 1.0, :gamma => 0.01, :degree => 3),
    Dict(:kernel => "sigmoid", :cost => 1.0, :gamma => 0.01, :coef0 => 0.0),
    =#
]

# Decision Tree Configurations (at least 6 different depth values)
dt_configs = [
    Dict(:max_depth => 3, :rng => rng),
    Dict(:max_depth => 5, :rng => rng),
    Dict(:max_depth => 7, :rng => rng),
    Dict(:max_depth => 10, :rng => rng),
    Dict(:max_depth => 15, :rng => rng),
    Dict(:max_depth => 20, :rng => rng),
]

# kNN Configurations (at least 6 different k values)
knn_configs = [
    Dict(:n_neighbors => 1),
    Dict(:n_neighbors => 3),
    Dict(:n_neighbors => 5),
    Dict(:n_neighbors => 7),
    Dict(:n_neighbors => 10),
    Dict(:n_neighbors => 15),
]

configs = Dict(
    :ANN => ann_configs,
    :SVM => svm_configs,
    :DT => dt_configs,
    :KNN => knn_configs
)

function run_approach_experiments(approach_name::String,
                                  model_configs::Dict,  
                                  train_inputs::Matrix{Float64}, 
                                  train_targets::Vector{String},
                                  test_inputs::Matrix{Float64},
                                  test_targets::Vector{String},
                                  k_folds::Int=5)
    results_df = DataFrame(
        Approach = String[],
        Model = Symbol[],
        Config = String[],
        Accuracy = Float64[],
        F1 = Float64[],
        Sensitivity = Float64[],
        Specificity = Float64[]   
    )

    ann_configs = get(configs, :ANN, [])
    svm_configs = get(configs, :SVM, [])
    dt_configs = get(configs, :DT, [])
    knn_configs = get(configs, :KNN, [])
    
    println("=" ^ 80)
    println("APPROACH: $approach_name")
    println("=" ^ 80)
    
    # Normalize training data
    normalizeMinMax!(train_inputs)
    norm_params = calculateMinMaxNormalizationParameters(train_inputs)
    
    # Normalize test data using training parameters
    test_inputs_norm = normalizeMinMax(test_inputs, norm_params)
    
    if (approach_name == "PCA")
        train_inputs = pcaToMatrix(train_inputs)
    end

    # Prepare dataset tuple
    train_inputs_f32 = Float32.(train_inputs)
    dataset = (train_inputs_f32, train_targets)
    
    # Generate cross-validation indices
    cv_indices = crossvalidation(train_targets, k_folds, rng)
    
    # ========================================================================
    # HYPERPARAMETER TESTING
    # ========================================================================
    
    println("\n--- Testing different model configurations ---")
    
    
    # Store best results
    best_ann = nothing
    best_ann_acc = -1.0
    best_svm = nothing
    best_svm_acc = -1.0
    best_dt = nothing
    best_dt_acc = -1.0
    best_knn = nothing
    best_knn_acc = -1.0
    
    # Test ANN
    println("\nTesting ANN configurations...")
    for (i, config) in enumerate(ann_configs)
        println("  ANN Config $i/$(length(ann_configs)): topology=$(config[:topology])")
        results = modelCrossValidation(:ANN, config, dataset, cv_indices)
        acc_mean = results[1][1]
        println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
        if acc_mean > best_ann_acc
            best_ann_acc = acc_mean
            best_ann = config
        end
        push!(results_df, (
            approach_name, 
            :ANN,
            string(config), 
            results[1][1],  # Mean Accuracy
            results[7][1],   # Mean F1
            results[3][1], # Mean Sensitivity
            results[4][1]  # Mean Specificity
        ))
    end
    

    
    
    # Test SVM
    println("\nTesting SVM configurations...")
    for (i, config) in enumerate(svm_configs)
        println("  SVM Config $i/$(length(svm_configs)): kernel=$(config[:kernel]), cost=$(config[:cost])")
        results = modelCrossValidation(:SVC, config, dataset, cv_indices)
        acc_mean = results[1][1]
        println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
        if acc_mean > best_svm_acc
            best_svm_acc = acc_mean
            best_svm = config
        end
        push!(results_df, (
            approach_name, 
            :SVM,
            string(config), 
            results[1][1],  # Mean Accuracy
            results[7][1],   # Mean F1
            results[3][1], # Mean Sensitivity
            results[4][1]  # Mean Specificity
        ))
    end
    
    # Test Decision Tree
    println("\nTesting Decision Tree configurations...")
    for (i, config) in enumerate(dt_configs)
        println("  DT Config $i/$(length(dt_configs)): max_depth=$(config[:max_depth])")
        results = modelCrossValidation(:DecisionTreeClassifier, config, dataset, cv_indices)
        acc_mean = results[1][1]
        println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
        if acc_mean > best_dt_acc
            best_dt_acc = acc_mean
            best_dt = config
        end
        push!(results_df, (
            approach_name, 
            :DT,
            string(config), 
            results[1][1],  # Mean Accuracy
            results[7][1],   # Mean F1
            results[3][1], # Mean Sensitivity
            results[4][1]  # Mean Specificity
        ))
    end
    
    
    # Test kNN
    println("\nTesting kNN configurations...")
    for (i, config) in enumerate(knn_configs)
        println("  kNN Config $i/$(length(knn_configs)): k=$(config[:n_neighbors])")
        results = modelCrossValidation(:KNeighborsClassifier, config, dataset, cv_indices)
        acc_mean = results[1][1]
        println("    Accuracy: $(round(acc_mean, digits=4)) ± $(round(results[1][2], digits=4))")
        if acc_mean > best_knn_acc
            best_knn_acc = acc_mean
            best_knn = config
        end
        push!(results_df, (
            approach_name, 
            :KNN,
            string(config), 
            results[1][1],  # Mean Accuracy
            results[7][1],   # Mean F1
            results[3][1], # Mean Sensitivity
            results[4][1]  # Mean Specificity
        ))
    end
    
    println("\n--- Best configurations found ---")
    #println("  Best ANN: $(best_ann[:topology]) - Accuracy: $(round(best_ann_acc, digits=4))")
    #println("  Best SVM: $(best_svm[:kernel]) (cost=$(best_svm[:cost])) - Accuracy: $(round(best_svm_acc, digits=4))")
    #println("  Best DT: max_depth=$(best_dt[:max_depth]) - Accuracy: $(round(best_dt_acc, digits=4))")
    println("  Best kNN: k=$(best_knn[:n_neighbors]) - Accuracy: $(round(best_knn_acc, digits=4))")
    #=
    # ========================================================================
    # ENSEMBLE TRAINING
    # ========================================================================
    
    println("\n--- Training ensemble model (combining best 3 models) ---")
    
    # Use best 3 models for ensemble (Stack ensemble combining multiple models)
    ensemble_estimators = [:SVC, :DecisionTreeClassifier, :KNeighborsClassifier]
    
    # Prepare training data for ensemble (needs Bool array)
    classes = unique(train_targets)
    train_targets_bool = oneHotEncoding(train_targets, classes)
    
    # The Stack ensemble function expects modelsHyperParameters to be a Dict
    # where each estimator's hyperparameters are stored
    # create_tuned_model will be called with modelsHyperParameters[estimator]
    # So we structure it as: modelsHyperParameters[estimator] = hyperparams_dict
    ensemble_hyperparams = Dict(
        :SVC => best_svm,
        :DecisionTreeClassifier => best_dt,
        :KNeighborsClassifier => best_knn
    )
    
    ensemble_config = Dict(:rng => rng)
    
    # Use the multi-estimator Stack ensemble
    println("Training Stack ensemble with 3 models (SVM, DT, kNN)...")
    ensemble_results = trainClassEnsemble(
        ensemble_estimators,
        ensemble_hyperparams,
        ensemble_config,
        (train_inputs, train_targets_bool),
        cv_indices
    )
    
    println("Ensemble Results:")
    println("  Accuracy: $(round(ensemble_results[1][1], digits=4)) ± $(round(ensemble_results[1][2], digits=4))")
    println("  F1-Score: $(round(ensemble_results[7][1], digits=4)) ± $(round(ensemble_results[7][2], digits=4))")
    push!(results_df, (
        approach_name, 
        :Stack,
        string(Dict(:ensemble => emsemble_config, :models => ensemble_hyperparams)), 
        results[1][1],  # Mean Accuracy
        results[7][1],   # Mean F1
        results[3][1], # Mean Sensitivity
        results[4][1]  # Mean Specificity
    ))
    =#
    
    return (results_df, Dict(
        :best_ann => best_ann,
        :best_svm => best_svm,
        :best_dt => best_dt,
        :best_knn => best_knn,
    ))
end

# ============================================================================
# APPROACH 1: Full Features Dataset
# ============================================================================

results_df, best_configs = run_approach_experiments(
    "Full Features Dataset",
    configs,
    copy(train_inputs),
    copy(train_targets),
    copy(test_inputs),
    copy(test_targets),
    3
)




results_df_pca, best_configs_pca = run_approach_experiments(
    "PCA",
    configs,
    copy(train_inputs),
    copy(train_targets),
    copy(test_inputs),
    copy(test_targets),
    3
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

println("\n" * "=" ^ 80)
println("FINAL SUMMARY - All Approaches")
println("=" ^ 80)

println("\n" * "=" ^ 80)
println("SUMMARY - All features")
println("=" ^ 80)
println(results_df)
println("=" ^ 80)
println("Best Configurations - All features")
println(best_configs)
println("=" ^ 80)

println("\n" * "=" ^ 80)
println("SUMMARY - PCA")
println("=" ^ 80)
println(results_df_pca)
println("=" ^ 80)
println("Best Configurations - PCA")
println(best_configs_pca)
println("=" ^ 80)

println("\n" * "=" ^ 80)
println("Pipeline completed successfully!")
println("=" ^ 80)

