using Random;
include("../mlj_models/models.jl")

"""
    calculateZeroMeanNormalizationParameters(dataset)

Compute the per-feature mean and standard deviation of a dataset.

# Arguments
- `dataset`: Matrix of real-valued features (observations in rows, features in columns).

# Returns
Tuple `(means, stds)` where each is a 1×D array containing the mean and standard
deviation of each feature column.
"""
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    dataset_means = mean(dataset, dims=1)
    dataset_stds = std(dataset, dims=1)

    return (dataset_means, dataset_stds)
end

"""
        normalizeZeroMean!(dataset, normalizationParameters)

In-place zero-mean, unit-variance normalization of a dataset.

Each feature column is centered and scaled using the provided means and
standard deviations. Columns with zero variance are set to zero.

# Arguments
- `dataset`: Matrix of real-valued features to normalize in place.
- `normalizationParameters`: Tuple `(means, stds)` as returned by
    `calculateZeroMeanNormalizationParameters`.
"""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    (vec_means, vec_stds) = normalizationParameters
    dataset .-= vec_means
    dataset ./= vec_stds
    
    dataset[:, vec(vec_stds) .== 0.0] .= 0.0
end

"""
    normalizeZeroMean!(dataset)

In-place zero-mean, unit-variance normalization using statistics computed from
`dataset` itself.

This is a convenience wrapper around `calculateZeroMeanNormalizationParameters`
and `normalizeZeroMean!`.
"""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    (vec_means, vec_stds) = calculateZeroMeanNormalizationParameters(dataset)
    dataset .-= vec_means
    dataset ./= vec_stds
    
    dataset[:, vec(vec_stds) .== 0.0] .= 0.0  
end

"""
    normalizeZeroMean(dataset, normalizationParameters)

Return a zero-mean, unit-variance normalized copy of `dataset`.

Normalization uses the given `(means, stds)` parameters and leaves the original
`dataset` unchanged.
"""
function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    (vec_means, vec_stds) = normalizationParameters
    
    dataset_zscore = (dataset .- vec_means) ./ vec_stds
    
    dataset_zscore[:, vec(vec_stds) .== 0.0] .= 0.0

    return dataset_zscore
end

"""
    normalizeZeroMean(dataset)

Return a zero-mean, unit-variance normalized copy of `dataset`, computing the
normalization parameters from `dataset` itself.
"""
function normalizeZeroMean( dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    
    (vec_means, vec_stds) = normalizationParameters
    dataset_zscore = (dataset .- vec_means) ./ vec_stds
    
    dataset_zscore[:, vec(vec_stds) .== 0.0] .= 0.0

    return dataset_zscore
end

"""
    holdOut(N, P[, rng])

Randomly split indices `1:N` into training and test sets.

# Arguments
- `N`: Total number of samples.
- `P`: Fraction of samples to assign to the test set (between 0 and 1).
- `rng`: Optional random number generator (default: `Random.default_rng()`).

# Returns
Tuple `(trainIndexes, testIndexes)` of integer index vectors.
"""
function holdOut(N::Int, P::Float64, rng::AbstractRNG=Random.default_rng())

    @assert 0 <= P <= 1 "P must be a value between 0 and 1.";

    indexes = randperm(rng, N)
    testSamples = round(Int, N * P)

    testIndexes = indexes[1:testSamples]
    trainIndexes = indexes[testSamples+1:end]

    @assert isempty(intersect(Set(testIndexes), Set(trainIndexes))) "The sets are not disjoint"
    @assert length(testIndexes)+length(trainIndexes) == N "The size of sets are not equal to N"


    return (trainIndexes, testIndexes)
end


"""
    holdOut(N, Pval, Ptest[, rng])

Randomly split indices `1:N` into training, validation and test sets.

# Arguments
- `N`: Total number of samples.
- `Pval`: Fraction of samples for the validation set.
- `Ptest`: Fraction of samples for the test set.
- `rng`: Optional random number generator.

# Returns
Tuple `(trainIndexes, validationIndexes, testIndexes)` of index vectors.
"""
function holdOut(N::Int, Pval::Float64, Ptest::Float64, rng::AbstractRNG=Random.default_rng())
    
    @assert (Pval + Ptest) < 1.0 "Pval and Ptest sum can't be greater than 1";

    validationAndTestPercentage = Pval + Ptest

    (trainIndexes, validationAndTestIndexes) = holdOut(N, validationAndTestPercentage, rng)

    validationAndTestSamples = length(validationAndTestIndexes)
    
    # Relative percentage of validation set respect of size of validationAndTestSamples
    if validationAndTestSamples > 0
        relativeValidationPercentage = Pval / validationAndTestPercentage
    else
        relativeValidationPercentage = 0
    end
    

    (temporalValidationIndexes, temporalTestIndexes) = holdOut(validationAndTestSamples, 1.0 - relativeValidationPercentage, rng)
    
    validationIndexes = validationAndTestIndexes[temporalValidationIndexes]
    testIndexes = validationAndTestIndexes[temporalTestIndexes]

    @assert isempty(intersect(Set(validationIndexes), Set(trainIndexes), Set(testIndexes))) "The sets are not disjoint"
    @assert length(validationIndexes)+length(trainIndexes)+length(testIndexes) == N "he size of sets are not equal to N"


    return (trainIndexes, validationIndexes, testIndexes)
end

"""
    calculateMinMaxNormalizationParameters(dataset)

Compute per-feature minimum and maximum values for min-max normalization.

# Arguments
- `dataset`: Matrix of real-valued features.

# Returns
Tuple `(mins, maxs)` with the minimum and maximum of each feature column.
"""
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

"""
    normalizeMinMax!(dataset, normalizationParameters)

In-place min-max normalization of `dataset` to the [0, 1] range.

Columns where `min == max` (no variation) are set to zero.

# Arguments
- `dataset`: Matrix of real-valued features to normalize in place.
- `normalizationParameters`: Tuple `(mins, maxs)` as returned by
  `calculateMinMaxNormalizationParameters`.
"""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
minValues = normalizationParameters[1];
maxValues = normalizationParameters[2];
dataset .-= minValues;
dataset ./= (maxValues .- minValues);
# eliminate any atribute that do not add information
dataset[:, vec(minValues.==maxValues)] .= 0;
return dataset;
end;

"""
    normalizeMinMax!(dataset)

In-place min-max normalization of `dataset` using parameters computed from the
data itself.
"""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

"""
    normalizeMinMax(dataset, normalizationParameters)

Return a min-max normalized copy of `dataset` using the provided parameters.
"""
function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
normalizeMinMax!(copy(dataset), normalizationParameters);
end;

"""
    normalizeMinMax(dataset)

Return a min-max normalized copy of `dataset` using parameters computed from
the data itself.
"""
function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;



# Accuracy

"""
    accuracy(outputs, targets)

Compute classification accuracy for binary labels represented as Boolean
vectors.
"""
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;


"""
    accuracy(outputs, targets)

Compute classification accuracy for one-hot (or single-column) Boolean output
and target matrices.

If there is a single output column, accuracy is computed elementwise. For
multi-column one-hot encodings, a prediction is considered correct if all
columns for a pattern match the target.
"""
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

"""
    accuracy(outputs, targets; threshold=0.5)

Compute accuracy given real-valued scores and Boolean targets, thresholding the
scores to Booleans.
"""
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
    threshold::Real=0.5)
accuracy(outputs.>=threshold, targets);
end;

"""
    accuracy(outputs, targets; threshold=0.5)

Compute accuracy for real-valued outputs and one-hot Boolean targets.

If there is a single output column, scores are thresholded directly. For
multi-column outputs, `classifyOutputs` is used to obtain Boolean predictions.
"""
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5)

    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

# CROSS validation

"""
    crossvalidation(N, k[, rng])

Assign each of `N` samples to one of `k` folds at random.

# Returns
An integer vector of length `N` with fold identifiers in `1:k`.
"""
function crossvalidation(N::Int64, k::Int64, rng::AbstractRNG=Random.default_rng())

    vector=collect(1:k)

    number_repetitions = ceil(Int, N/k)

    vector = repeat(vector, number_repetitions)

    return shuffle!(rng, vector[1:N])
end


"""
    crossvalidation(targets::AbstractArray{Bool,1}, k[, rng])

Create stratified `k`-fold indices for binary targets stored as a Boolean
vector.

Positive and negative samples are assigned to folds separately to preserve
class balance.
"""
function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int64, rng::AbstractRNG=Random.default_rng())

    indices_vector = zeros(Int, size(targets,1))

    positive_indexes = findall(t -> t, targets)
    negative_indexes = findall(t -> !t, targets)

    indices_vector[positive_indexes] = crossvalidation(size(positive_indexes,1), k, rng)
    indices_vector[negative_indexes] = crossvalidation(size(negative_indexes,1), k, rng)

    return indices_vector;

end

"""
    crossvalidation(targets::AbstractArray{Bool,2}, k[, rng])

Create stratified `k`-fold indices for multi-class one-hot Boolean targets.
"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64, rng::AbstractRNG=Random.default_rng())

  indices_vector = zeros(Int, size(targets,1))

  [indices_vector[findall(targets[:,i])] = crossvalidation(sum(targets[:, i]), k, rng) for i in 1:size(targets,2)]

  return indices_vector

end


"""
    crossvalidation(targets, k[, rng])

Create stratified `k`-fold indices for arbitrary categorical targets by first
applying one-hot encoding.
"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64, rng::AbstractRNG=Random.default_rng())
    result =  crossvalidation(oneHotEncoding(targets), k, rng)
    @assert size(targets) == size(result)
    return result;
  end


## ONE HOT ENCODING
    """
            oneHotEncoding(feature, classes)

    Convert a categorical feature vector to a one-hot encoded Boolean matrix.

    For binary problems, a single-column representation is used; otherwise a
    column per class is created.
    """
    function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})

    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));

    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)

    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
    
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    
    end;
    return oneHot;
end;

"""
    oneHotEncoding(feature)

One-hot encode `feature` using its unique values as classes.
"""
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

"""
    oneHotEncoding(feature::AbstractArray{Bool,1})

Treat a Boolean feature as a binary one-hot encoded column vector.
"""
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);


## CLASSIFY OUTPUTS

"""
    classifyOutputs(outputs; threshold=0.5)

Convert real-valued output scores to Boolean one-hot predictions.

For a single output, scores are thresholded. For multiple outputs, the maximum
score per instance is selected and marked as `true`.
"""
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
    threshold::Real=0.5) 
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

"""
    pcaToMatrix(train_inputs)

Fit a PCA model (via `getPCA`) on `train_inputs` and return the transformed
features as a dense matrix.
"""
function pcaToMatrix(train_inputs::Matrix{Float64})
    pca_model = getPCA()
    pca_mach = machine(pca_model, MLJ.table(train_inputs))
    MLJ.fit!(pca_mach)

    # Transform the data
    pca_train = MLJ.transform(pca_mach, MLJ.table(train_inputs))
    #pca_test = MLJ.transform(pca_mach, MLJ.table(test_inputs))
    subarrays = values(pca_train)
    matrix_result = hcat(map(subarray -> transpose(Matrix(transpose(subarray))), subarrays)...)
    return matrix_result
end


