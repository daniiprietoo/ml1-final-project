using MLJ
using LIBSVM
using NearestNeighborModels
using DecisionTree
using Random

# SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
# kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
# DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

SVC = @load ProbabilisticSVC pkg=LIBSVM
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


# modelSVMClassifier = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=2.0, degree=Int32(3))

# modelDTClassifier = DTClassifier(max_depth=4, rng=Random.MersenneTwister(1))

# modelknnClassifier = kNNClassifier(K=3)

# modelSVMClassifier = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=2.0, degree=Int32(3))

modelDTClassifier = DTClassifier(max_depth=4, rng=Random.MersenneTwister(1))

modelknnClassifier = kNNClassifier(K=3)


function getSVCModel(modelHyperparameters::Dict)

    kernelSelected = get(modelHyperparameters, :kernel, "sigmoid")
    @assert kernelSelected in ["linear","rbf", "sigmoid", "poly"] "Kernel not supported"

    gamma=get(modelHyperparameters, :gamma, 0.1)
    degree=get(modelHyperparameters, :degree, 3)
    coef0=get(modelHyperparameters, :coef0, 0.0)
    cost=get(modelHyperparameters, :cost, 1.0)

    if kernelSelected == "linear"
        kernel= LIBSVM.Kernel.Linear
        return SVMClassifier(kernel=kernel, cost=Float64(1.0))
    elseif kernelSelected == "rbf"
        kernel= LIBSVM.Kernel.RadialBasis
        return SVMClassifier(kernel=kernel, cost=Float64(1.0), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    elseif kernelSelected == "sigmoid"
        kernel= LIBSVM.Kernel.Sigmoid
        return SVMClassifier(kernel=kernel, gamma=Float64(gamma), coef0=Float64(coef0))
    elseif kernelSelected == "poly"
        kernel= LIBSVM.Kernel.Polynomial
        return SVMClassifier(kernel=kernel, gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    end
end

function getDecisionTreeModel(modelHyperparameters::Dict)
    max_depth=get(modelHyperparameters, :max_depth, 5)
    rng=get(modelHyperparameters, :rng, Random.MersenneTwister(1))
    return modelDTClassifier = DTClassifier(max_depth=max_depth, rng=rng)
end

function getkNNModel(modelHyperparameters::Dict)
    n_neighbors=get(modelHyperparameters, :n_neighbors, 3)
    return kNNClassifier(K=n_neighbors)
end

function getSVCProbabilisticModel(modelHyperparameters::Dict)

    kernelSelected = get(modelHyperparameters, :kernel, "sigmoid")
    @assert kernelSelected in ["linear","rbf", "sigmoid", "poly"] "Kernel not supported"

    gamma=get(modelHyperparameters, :gamma, 0.1)
    degree=get(modelHyperparameters, :degree, 3)
    coef0=get(modelHyperparameters, :coef0, 0.0)
    cost=get(modelHyperparameters, :cost, 1.0)

    if kernelSelected == "linear"
        kernel= LIBSVM.Kernel.Linear
        return SVC(kernel=kernel, cost=Float64(1.0))
    elseif kernelSelected == "rbf"
        kernel= LIBSVM.Kernel.RadialBasis
        return SVC(kernel=kernel, cost=Float64(1.0), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    elseif kernelSelected == "sigmoid"
        kernel= LIBSVM.Kernel.Sigmoid
        return SVC(kernel=kernel, gamma=Float64(gamma), coef0=Float64(coef0))
    elseif kernelSelected == "poly"
        kernel= LIBSVM.Kernel.Polynomial
        return SVC(kernel=kernel, gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    end
end

function getDecisionTreeModel(modelHyperparameters::Dict)
    max_depth=get(modelHyperparameters, :max_depth, 5)
    rng=get(modelHyperparameters, :rng, Random.MersenneTwister(1))
    return modelDTClassifier = DTClassifier(max_depth=max_depth, rng=rng)
end

function getkNNModel(modelHyperparameters::Dict)
    n_neighbors=get(modelHyperparameters, :n_neighbors, 3)
    return kNNClassifier(K=n_neighbors)
end

function getModel(modelType::Symbol, modelHyperparameters::Dict)

    @assert modelType in (:SVC, :DecisionTreeClassifier, :KNeighborsClassifier, :SVCProbabilistic) "Only SVC, DecisionTreeClassifier and KNN are supported"

    model = nothing
    if modelType == :SVC
        model = getSVCModel(modelHyperparameters)
    elseif modelType == :SVCProbabilistic
            model = getSVCProbabilisticModel(modelHyperparameters)
    elseif modelType == :DecisionTreeClassifier
        model = getDecisionTreeModel(modelHyperparameters)
    elseif modelType == :KNeighborsClassifier
        model = getkNNModel(modelHyperparameters)
    end


    return model


end


