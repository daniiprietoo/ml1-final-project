using MLJ
using LIBSVM
using NearestNeighborModels
using DecisionTree
using Random

# SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
# kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
# DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

SVC = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
CatBoostClassifier = @load CatBoostClassifier pkg=CatBoost verbosity=0
RFClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=0
AdaBoostClassifier = @load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
CatBoostClassifier = @load CatBoostClassifier pkg=CatBoost verbosity=0
DTRegressor = MLJ.@load DecisionTreeRegressor pkg=DecisionTree verbosity=0
LinearSVC = @load LinearSVC pkg=LIBSVM
PCA = MLJ.@load PCA pkg=MultivariateStats verbosity=0
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
# modelSVMClassifier = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=2.0, degree=Int32(3))

# modelDTClassifier = DTClassifier(max_depth=4, rng=Random.MersenneTwister(1))

# modelknnClassifier = kNNClassifier(K=3)

# modelSVMClassifier = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=2.0, degree=Int32(3))


function getSVCModel(modelHyperparameters::Dict)

    kernelSelected = get(modelHyperparameters, :kernel, "sigmoid")
    @assert kernelSelected in ["linear","rbf", "sigmoid", "poly"] "Kernel not supported"

    gamma=get(modelHyperparameters, :gamma, 0.1)
    degree=get(modelHyperparameters, :degree, 3)
    coef0=get(modelHyperparameters, :coef0, 0.0)
    cost=get(modelHyperparameters, :cost, 1.0)

    if kernelSelected == "linear"
        return LinearSVC(cost = Float64(cost))
    elseif kernelSelected == "rbf"
        kernel= LIBSVM.Kernel.RadialBasis
        return SVMClassifier(kernel=kernel, cost=Float64(cost), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    elseif kernelSelected == "sigmoid"
        kernel= LIBSVM.Kernel.Sigmoid
        return SVMClassifier(kernel=kernel, gamma=Float64(gamma), coef0=Float64(coef0))
    elseif kernelSelected == "poly"
        kernel= LIBSVM.Kernel.Polynomial
        return SVMClassifier(kernel=kernel, gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    end
end

# function getDecisionTreeModel(modelHyperparameters::Dict)
#     max_depth=get(modelHyperparameters, :max_depth, 5)
#     rng=get(modelHyperparameters, :rng, Random.MersenneTwister(1))
#     return modelDTClassifier = DTClassifier(max_depth=max_depth, rng=rng)
# end

function getLR()
    return LogisticClassifier()
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
        return SVC(kernel=kernel, cost=Float64(cost))
    elseif kernelSelected == "rbf"
        kernel= LIBSVM.Kernel.RadialBasis
        return SVC(kernel=kernel, cost=Float64(cost), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    elseif kernelSelected == "sigmoid"
        kernel= LIBSVM.Kernel.Sigmoid
        return SVC(kernel=kernel, gamma=Float64(gamma), coef0=Float64(coef0))
    elseif kernelSelected == "poly"
        kernel= LIBSVM.Kernel.Polynomial
        return SVC(kernel=kernel, gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))
    end
end

function getDecisionTreeModel(modelHyperparameters::Dict)
    max_depth = get(modelHyperparameters, :max_depth, 5)
    min_samples_leaf = get(modelHyperparameters, :min_samples_leaf, 1)
    rng = get(modelHyperparameters, :rng, Random.MersenneTwister(1234))
    return DTClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, rng=rng)
end
function getDecisionTreeModelRegressor(modelHyperparameters::Dict)
    max_depth=get(modelHyperparameters, :max_depth, 5)
    rng=get(modelHyperparameters, :rng, Random.MersenneTwister(1234))
    return DTRegressor(max_depth=max_depth, rng=rng)
end

function getkNNModel(modelHyperparameters::Dict)
    n_neighbors=get(modelHyperparameters, :n_neighbors, 3)
    return kNNClassifier(K=n_neighbors)
end

function getAdaBoostModel(modelHyperparameters::Dict)
    n_iter = get(modelHyperparameters, :n_estimators, 50)
    rng = get(modelHyperparameters, :rng, Random.MersenneTwister(1))
    return AdaBoostClassifier(n_iter=n_iter, rng=rng)
end

function getRandomForestModel(modelHyperparameters::Dict)
    n_trees = get(modelHyperparameters, :n_trees, 100)
    max_depth = get(modelHyperparameters, :max_depth, -1)  # -1 means no limit
    rng = get(modelHyperparameters, :rng, Random.MersenneTwister(1234))
    return RFClassifier(n_trees=n_trees, max_depth=max_depth, rng=rng)
end

function getCatBoostModel(modelHyperparameters::Dict)
    iterations = get(modelHyperparameters, :iterations, 100)
    learning_rate = get(modelHyperparameters, :learning_rate, 0.1)
    depth = get(modelHyperparameters, :depth, 6)
    return CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth)
end


function getPCA()
    return PCA(variance_ratio=0.95)
end

function getModel(modelType::Symbol, modelHyperparameters::Dict)

    # @assert modelType in (:SVC, :DecisionTreeClassifier, :KNeighborsClassifier, :SVCProbabilistic) "Only SVC, DecisionTreeClassifier and KNN are supported"

    model = nothing
    if modelType == :SVC
        model = getSVCModel(modelHyperparameters)
    elseif modelType == :SVCProbabilistic
            model = getSVCProbabilisticModel(modelHyperparameters)
    elseif modelType == :DecisionTreeClassifier
        model = getDecisionTreeModel(modelHyperparameters)
    elseif modelType == :KNeighborsClassifier
        model = getkNNModel(modelHyperparameters)
    elseif modelType == :RandomForestClassifier
        model = getRandomForestModel(modelHyperparameters)
    elseif modelType == :AdaBoostClassifier
        model = getAdaBoostModel(modelHyperparameters)
    elseif modelType == :CatBoostClassifier
        model = getCatBoostModel(modelHyperparameters)
    elseif modelType == :LinearRegressor
        model = getLR()
    else
        error("Tipo de modelo no soportado: ", modelType)
    end


    return model
end




