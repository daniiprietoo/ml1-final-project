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

include("code/general/run_approach.jl")

println("=" ^ 80)
println("Random seed set to: $RANDOM_SEED for reproducibility")
println()

# DATA LOADING AND PREPROCESSING

println("Step 1: Loading and preprocessing data...")
const TRACKS_FILE = "data/tracks.csv"
const FEATURES_FILE = "data/features.csv"

# Load and merge data
df = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens])#[1:10000, :]
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
    Dict(:topology => [10], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [20], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [30], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [50], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [10, 5], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [20, 10], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [30, 15], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
    Dict(:topology => [50, 25], :learningRate => 0.01, :maxEpochs => 1000, :validationRatio => 0.2, :maxEpochsVal => 20, :numExecutions => 3),
]

# SVM Configurations (at least 8 different configurations: kernels + C values)
svm_configs = [
    
    Dict(:kernel => "linear", :cost => 0.1),
    Dict(:kernel => "linear", :cost => 1.0),
    Dict(:kernel => "rbf", :cost => 0.1, :gamma => 0.01),
    Dict(:kernel => "rbf", :cost => 1.0, :gamma => 0.01),
    Dict(:kernel => "rbf", :cost => 10.0, :gamma => 0.1),
    Dict(:kernel => "poly", :cost => 1.0, :gamma => 0.01, :degree => 2),
    Dict(:kernel => "poly", :cost => 1.0, :gamma => 0.01, :degree => 3),
    Dict(:kernel => "sigmoid", :cost => 1.0, :gamma => 0.01, :coef0 => 0.0),
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

# ============================================================================
# APPROACH 1: Full Features Dataset
# ============================================================================

results_df, best_configs = run_approach_experiments(
    "Full Features Dataset",
    configs,
    copy(train_inputs),
    copy(train_targets),
    copy(test_inputs),
    copy(test_targets);
    k_folds=3,
    rng=rng,
)


results_df_pca, best_configs_pca = run_approach_experiments(
    "PCA",
    configs,
    copy(train_inputs),
    copy(train_targets),
    copy(test_inputs),
    copy(test_targets),
    k_folds=3,
    rng=rng,
    preprocessing=Dict(:type => :PCA, :maxoutdim => 0.97)
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

