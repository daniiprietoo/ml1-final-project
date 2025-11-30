using Random;
include("../mlj_models/models.jl")

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

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

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

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;



# Accuracy

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
    threshold::Real=0.5)
accuracy(outputs.>=threshold, targets);
end;

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

function crossvalidation(N::Int64, k::Int64, rng::AbstractRNG=Random.default_rng())

    vector=collect(1:k)

    number_repetitions = ceil(Int, N/k)

    vector = repeat(vector, number_repetitions)

    return shuffle!(rng, vector[1:N])
end


function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int64, rng::AbstractRNG=Random.default_rng())

    indices_vector = zeros(Int, size(targets,1))

    positive_indexes = findall(t -> t, targets)
    negative_indexes = findall(t -> !t, targets)

    indices_vector[positive_indexes] = crossvalidation(size(positive_indexes,1), k, rng)
    indices_vector[negative_indexes] = crossvalidation(size(negative_indexes,1), k, rng)

    return indices_vector;

end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64, rng::AbstractRNG=Random.default_rng())

  indices_vector = zeros(Int, size(targets,1))

  [indices_vector[findall(targets[:,i])] = crossvalidation(sum(targets[:, i]), k, rng) for i in 1:size(targets,2)]

  return indices_vector

end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64, rng::AbstractRNG=Random.default_rng())
    result =  crossvalidation(oneHotEncoding(targets), k, rng)
    @assert size(targets) == size(result)
    return result;
  end


## ONE HOT ENCODING
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

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);


## CLASSIFY OUTPUTS

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


