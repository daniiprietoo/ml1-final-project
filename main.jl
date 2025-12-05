using Random
using Statistics
using MLJ

# Set random seed for reproducibility
const RANDOM_SEED = 1234
Random.seed!(RANDOM_SEED)
const rng = MersenneTwister(RANDOM_SEED)
MLJ.default_resource(CPUProcesses())

# Include all necessary modules
include("code/general/preprocessing.jl")
include("code/general/run_approach.jl")
include("code/general/utils_plot.jl")
println("=" ^ 80)
println("Random seed set to: $RANDOM_SEED for reproducibility")
println()

# ============================================================================
# DATA LOADING AND PREPROCESSING (CORRECTED)
# ============================================================================

println("Step 1: Loading and preprocessing data...")
const TRACKS_FILE = "data/tracks.csv"
const FEATURES_FILE = "data/features.csv"

# Load data
df = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens])[1:30000, :]
reduced_df = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens], selected_features_algorithms=[:tonnetz, :mfcc, :spectral_centroid, :spectral_bandwith])[1:30000, :]

# Extract Raw Inputs (Features) and Raw Targets (Listens)
features_df = extract_features(df)
reduced_features_df = extract_features(reduced_df)
print(names(reduced_features_df))
inputs = Matrix{Float64}(features_df)
reduced_inputs = Matrix{Float64}(reduced_features_df)
listens = df.track_listens

println("Input features shape: $(size(inputs))")
println("REDUCE Input features shape: $(size(reduced_inputs))")

# ============================================================================
# TRAIN/TEST SPLIT (Perform Split FIRST to avoid Leakage)
# ============================================================================

println("Step 2: Performing train/test split...")
const TEST_RATIO = 0.2
(trainIndexes, testIndexes) = holdOut(size(inputs, 1), TEST_RATIO, rng)

# 1. Split the raw listen counts
train_listens = listens[trainIndexes]
test_listens = listens[testIndexes]

# 2. Calculate thresholds on Training Data
p33 = quantile(train_listens, 0.33)
p50 = quantile(train_listens, 0.5)
p66 = quantile(train_listens, 0.66)

println("Class Thresholds (calculated on Train only):")
println("  Low/Med Boundary (p33): $p33")
println("  Med Boundary (p50): $p50")
println("  Med/High Boundary (p66): $p66")

# 3. Define mapping functions using fixed thresholds
function create_popularity_class(listen_count)
    if listen_count <= p33
        return "Low"
    elseif listen_count <= p66
        return "Medium"
    else
        return "High"
    end
end

function binary_popularity_class(listen_count)
    if listen_count <= p50
        return "Low"
    else 
        return "High"
    end
end

# 4. Generate Targets
train_targets = [create_popularity_class(l) for l in train_listens]
test_targets = [create_popularity_class(l) for l in test_listens]

train_binary_targets = [binary_popularity_class(l) for l in train_listens]
test_binary_targets = [binary_popularity_class(l) for l in test_listens]

# 5. Split Inputs
train_inputs = inputs[trainIndexes, :]
test_inputs = inputs[testIndexes, :]

reduced_train_inputs = reduced_inputs[trainIndexes, :]
reduced_test_inputs = reduced_inputs[testIndexes, :]

println("Training set: $(size(train_inputs, 1)) samples")
println("Test set: $(size(test_inputs, 1)) samples")
println()

# Verify Class Distribution
println("Training Popularity Distribution:")
println("  Low: $(sum(train_targets .== "Low"))")
println("  Medium: $(sum(train_targets .== "Medium"))")
println("  High: $(sum(train_targets .== "High"))")
println()

println("Training Popularity Distribution (Binary):")
println("  Low: $(sum(train_binary_targets .== "Low"))")
println("  High: $(sum(train_binary_targets .== "High"))")
println()


# ============================================================================
# HELPER FUNCTION: Run experiments for an approach
# ============================================================================

# ANN Configurations (at least 8 different architectures, 1-2 hidden layers)
ann_configs = [
    Dict(:topology => [128], :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [64],  :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [128, 64], :learningRate => 0.001, :maxEpochs => 200, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [64, 32],  :learningRate => 0.001, :maxEpochs => 200, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [10, 5], :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [32, 16],  :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [32], :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
    Dict(:topology => [16], :learningRate => 0.001, :maxEpochs => 300, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 1),
]

# SVM Configurations (at least 8 different configurations: kernels + C values)
svm_configs = [
    Dict(:kernel => "linear", :cost => 0.1),
    Dict(:kernel => "linear", :cost => 1.0),
    Dict(:kernel => "linear", :cost => 10.0),
    Dict(:kernel => "linear", :cost => 20.0),
    Dict(:kernel => "linear", :cost => 50.0),
    Dict(:kernel => "linear", :cost => 100.0),
    Dict(:kernel => "sigmoid", :cost => 1.0, :gamma => 0.01, :coef0 => 0.0),
    Dict(:kernel => "sigmoid", :cost => 3.0, :gamma => 0.02, :coef0 => 0.1),
]

# Decision Tree Configurations (at least 6 different depth values)
dt_configs = [
    Dict(:max_depth => 25, :min_samples_leaf => 20, :rng => rng),
    Dict(:max_depth => 30, :min_samples_leaf => 20, :rng => rng),
    Dict(:max_depth => 7, :min_samples_leaf => 20, :rng => rng),
    Dict(:max_depth => 10, :min_samples_leaf => 20, :rng => rng),
    Dict(:max_depth => 15, :min_samples_leaf => 20, :rng => rng),
    Dict(:max_depth => 20, :min_samples_leaf => 20, :rng => rng),
]

# kNN Configurations (at least 6 different k values)
knn_configs = [
    Dict(:n_neighbors => 50),
    Dict(:n_neighbors => 25),
    Dict(:n_neighbors => 5),
    Dict(:n_neighbors => 7),
    Dict(:n_neighbors => 10),
    Dict(:n_neighbors => 15),
]
rf_configs = [
    Dict(:n_trees => 50, :max_depth => 5, :rng => rng),
    Dict(:n_trees => 100, :max_depth => 10, :rng => rng),
    Dict(:n_trees => 150, :max_depth => 15, :rng => rng),
    Dict(:n_trees => 100, :max_depth => -1, :rng => rng),
]

adaboost_configs = [
    Dict(:n_estimators => 25, :learning_rate => 0.5, :rng => rng),
    Dict(:n_estimators => 50, :learning_rate => 1.0, :rng => rng),
    Dict(:n_estimators => 100, :learning_rate => 1.0, :rng => rng),
    Dict(:n_estimators => 50, :learning_rate => 0.5, :rng => rng),
]

catboost_configs = [
    Dict(:iterations => 20, :learning_rate => 0.1, :depth => 4),
    Dict(:iterations => 40, :learning_rate => 0.1, :depth => 6),
    Dict(:iterations => 60, :learning_rate => 0.05, :depth => 8),
]

configs = Dict(
    :ANN => ann_configs,
    :SVM => svm_configs,
    :DT => dt_configs,
    :KNN => knn_configs,
    :RF => rf_configs,
    :AdaBoost => adaboost_configs,
    :CatBoost => catboost_configs
)

# ============================================================================
# APPROACH 1: Full Features Dataset
# ============================================================================

# results_df, best_configs = run_approach_experiments(
#     "Binary with Feature Reduction",
#     configs,
#     reduced_train_inputs,
#     train_binary_targets,
#     reduced_test_inputs,
#     test_binary_targets;
#     k_folds=3,
#     rng=rng,
#     normalize=:zero,
#     test_n=1
# )
# println("\n" * "=" ^ 80)
# println("SUMMARY - Binary with Feature Reduction")
# println("=" ^ 80)
# println(results_df)
# println("=" ^ 80)
# println("Best Configurations - Binary with Feature Reduction")
# println(best_configs)
# println("=" ^ 80)
# plot_grouped_comparison(results_df; title_str="Best Model Performance (Full): Accuracy vs F1")
# plot_tradeoff_scatter(results_df; title_str="Full Features: Trade-off Analysis")
# save_results_to_csv(results_df, "results/full_dataset.csv")


results_df_pca, best_configs_pca = run_approach_experiments(
    "Binary with PCA",
    configs,
    train_inputs,
    train_binary_targets,
    test_inputs,
    test_binary_targets,
    k_folds=3,
    rng=rng,
    preprocessing=Dict(:type => :PCA, :variance_ratio => 0.7),
    normalize=:zero,
    test_n=2
)


println("\n" * "=" ^ 80)
println("SUMMARY - Binary with PCA")
println("=" ^ 80)
println(results_df_pca)
println("=" ^ 80)
println("Best Configurations - Binary with PCA")
println(best_configs_pca)
println("=" ^ 80)

plot_grouped_comparison(results_df_pca; title_str="Best Model Performance (Bin PCA): Accuracy vs F1")
plot_tradeoff_scatter(results_df_pca; title_str="PCA Approach: Trade-off Analysis")
save_results_to_csv(results_df_pca, "results/pca.csv")

results_df_lda, best_configs_lda = run_approach_experiments(
    "LDA",
    configs,
    train_inputs,
    train_targets,
    test_inputs,
    test_targets,
    k_folds=3,
    rng=rng,
    preprocessing=Dict(:type => :LDA, :outdim => 2),
    normalize=:zero,
    test_n=3
)

println("\n" * "=" ^ 80)
println("SUMMARY - LDA")
println("=" ^ 80)
println(results_df_lda)
println("=" ^ 80)
println("Best Configurations - LDA")
println(best_configs_lda)
println("=" ^ 80)
plot_grouped_comparison(results_df_lda; title_str="Best Model Performance (LDA): Accuracy vs F1")
plot_tradeoff_scatter(results_df_lda; title_str="LDA Approach: Trade-off Analysis")
save_results_to_csv(results_df_lda, "results/lda.csv")

results_df_reduced, best_configs_reduced = run_approach_experiments(
    "Feature Reduction",
    configs,
    reduced_train_inputs,
    train_targets,
    reduced_test_inputs,
    test_targets,
    k_folds=3,
    rng=rng,
    normalize=:zero,
    test_n=4
)

println("\n" * "=" ^ 80)
println("SUMMARY - Feature Reduction")
println("=" ^ 80)
println(results_df_reduced)
println("=" ^ 80)
println("Best Configurations - Feature Reduction")
println(best_configs_reduced)
println("=" ^ 80)
plot_grouped_comparison(results_df_reduced; title_str="Best Model Performance (3-Class Reduced): Accuracy vs F1")
plot_tradeoff_scatter(results_df_reduced; title_str="Reduced 3-class Approach: Trade-off Analysis")
save_results_to_csv(results_df_reduced, "results/reduced.csv")


println("\n" * "=" ^ 80)
println("Pipeline completed successfully!")
println("=" ^ 80)

