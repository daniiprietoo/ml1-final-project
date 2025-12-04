## Model cross validation
include("../mlj_models/train_mlj.jl")
include("../ann/build_train.jl")
include("../mlj_models/models.jl")
using MLJEnsembles: EnsembleModel, CPUThreads


  function modelCrossValidation(
    modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1})


    if(modelType == :ANN)


        topology = get(modelHyperparameters, :topology, [4,3])
        transferFunctions = get(modelHyperparameters, :transferFunctions, fill(σ, length(topology)))
        learningRate = get(modelHyperparameters, :learningRate, 0.01)
        maxEpochs = get(modelHyperparameters, :maxEpochs, 1000)
        validationRatio = get(modelHyperparameters, :validationRatio, 0)
        maxEpochsVal = get(modelHyperparameters, :maxEpochsVal, 20)
        numExecutions = get(modelHyperparameters, :numExecutions, 20)

    
        return ANNCrossValidation(topology, dataset, crossValidationIndices;
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, 
        learningRate=learningRate,
        validationRatio=validationRatio, 
        maxEpochsVal=maxEpochsVal,
        numExecutions=numExecutions)


    elseif modelType in [:SVC, :DecisionTreeClassifier, :KNeighborsClassifier, :RandomForestClassifier, :AdaBoostClassifier, :CatBoostClassifier]
        # Los modelos de MLJ son deterministas, usamos la nueva función
        return mljCrossValidation(modelType, modelHyperparameters, dataset, crossValidationIndices)
    else
        error("Tipo de modelo desconocido: ", modelType)
    end

end

##TRAIN ENSEMBLE MODELS
function trainClassEnsemble(estimator::Symbol, 
    modelsHyperParameters::Dict,
    ensembleHyperParameters::Dict,     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},    
    kFoldIndices::Array{Int64,1})

    numFolds = maximum(kFoldIndices)
    results_vector = Vector{Dict{Symbol, Float64}}(undef, numFolds)
    input_data, output_data = trainingDataset

    for k in 1:numFolds

        println("Execute fold $k")

        test_indexes = (kFoldIndices .== k)
        train_indexes = .!test_indexes

        X_train_no_coerce = MLJ.table(input_data[train_indexes, :])
        X_train = coerce(X_train_no_coerce,  autotype(X_train_no_coerce, rules = (:discrete_to_continuous,)))
        y_train = categorical(output_data[train_indexes])
        
        X_test_no_coerce = MLJ.table(input_data[test_indexes, :])
        X_test = coerce(X_test_no_coerce, autotype(X_test_no_coerce, rules = (:discrete_to_continuous,)))
        y_test = categorical(output_data[test_indexes])

        model =  getModel(estimator, modelsHyperParameters)


        model_bagging= EnsembleModel(
            model = model,                                             
            bagging_fraction = get(ensembleHyperParameters, :baggin_fraction, 0.5),    
            rng =  get(ensembleHyperParameters, :rng, 123),                  
            acceleration = CPUThreads() 
        )

        #We define a range with lower 10 and upper 100
        #With grid 10 to test 10 values for this range (10,20,30..)
        #Holdout - like previous notebooks
        #measure log_loss - take into account how confident a model was in making its prediction 
        range_n = range(model_bagging, :n, lower=10, upper=100)

        tuned_ensemble = TunedModel(
            model=model_bagging,
            resampling=Holdout(fraction_train=0.8, shuffle=true),
            tuning=Grid(resolution=10),
            range=range_n,
            measure=log_loss # Puedes usar log_loss, accuracy, etc.
        )


        mach = machine(tuned_ensemble, X_train, y_train) |> MLJ.fit!

        predictions_prob = MLJ.predict(mach, X_test)
        

        #Convert probabilities in labels and convert it in labels
        predicted_labels = mode.(predictions_prob)
        predicted_labels_str = string.(predicted_labels)

        #Calculate the confusion matrix
        metrics = confusionMatrix(predicted_labels_str, string.(y_test))

   
        println("metrics for fold $k: $metrics")

        fold_results = Dict(
            :accuracy => metrics.accuracy,
            :error_rate => metrics.error_rate,
            :sensitivity => metrics.sensitivity,
            :specificity => metrics.specificity,
            :ppv => metrics.positive_predictive_value,
            :npv => metrics.negative_predictive_value,
            :f1_score => metrics.f_score
        ) 

        results_vector[k] = fold_results


    end

    #Define dict to return metric (mean and standard desviation) 
    aggregated_results = Dict{Symbol, Tuple{Float64, Float64}}()

    #Iterate over the metric of first fold to get the names (all folds have the same metrics)
    for metric_name in keys(results_vector[1])
        metric_values = [fold[metric_name] for fold in results_vector]
        
        mean_val = mean(metric_values)
        std_val = std(metric_values)
        
        aggregated_results[metric_name] = (mean_val, std_val)
        
        println("$(rpad(metric_name, 12)): Mean value: $(round(mean_val, digits=3)) - STD: $(round(std_val, digits=3))")
    end

    return (
        aggregated_results[:accuracy],
        aggregated_results[:error_rate],
        aggregated_results[:sensitivity],
        aggregated_results[:specificity],
        aggregated_results[:ppv],
        aggregated_results[:npv],
        aggregated_results[:f1_score]
    )
end


#Function to create tunned model, to optimize cost and max_depth in SVC and TreeClassifier
#Receive the model symbol, and if it is a tuned params defined for this model, returns a tuned mode.
#Otherwise, returns the base model
function create_tuned_model(estimator_symbol::Symbol, modelsHyperParameters::Dict)

    tuning_params_dict = Dict(
        :SVC => (
            param_symbol = :cost,
            lower = 0.1,
            upper = 10,
            scale = :log
        ),
        :DecisionTreeClassifier => (
            param_symbol = :max_depth,
            lower = 2,
            upper = 10,
            scale = :linear
        )
    )

    # For SVC in ensembles, use probabilistic version
    if estimator_symbol == :SVC
        base_model = getSVCProbabilisticModel(modelsHyperParameters)
    else
        base_model = getModel(estimator_symbol, modelsHyperParameters)
    end

    if haskey(tuning_params_dict, estimator_symbol)
        println("Creando TunedModel para $estimator_symbol...")
        
        tuning_params = tuning_params_dict[estimator_symbol]

        param_range = range(base_model, tuning_params.param_symbol; lower=tuning_params.lower, upper=tuning_params.upper, scale=tuning_params.scale)
        
        return TunedModel(
            model=base_model,
            resampling=Holdout(fraction_train=0.7),
            tuning=Grid(resolution=5),
            range=param_range,
            measure=log_loss
        )
    else
        println("No tuning params for $estimator_symbol. Using base mode.")
        return base_model
    end
end


function trainClassEnsemble(estimators::AbstractArray{Symbol, 1}, 
    modelsHyperParameters::Dict,
    ensembleHyperParameters::Dict,     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any, 1}},    
    kFoldIndices::Array{Int64,1},
    rng::AbstractRNG=MersenneTwister(1234))
    
    
    numFolds = maximum(kFoldIndices)

    results_vector = Vector{Dict{Symbol, Float64}}(undef, numFolds)

    input_data, output_data = trainingDataset
    targets_str = string.(output_data)
    classes = unique(targets_str)
    numClasses = length(classes)
    
    # Initialize total confusion matrix (using Float32 to match your DataFrame definition)
    total_confusion_matrix = zeros(Float32, numClasses, numClasses)

    for k in 1:numFolds

        println("Execute fold $k")

        test_indexes = (kFoldIndices .== k)
        train_indexes = .!test_indexes

        X_train_no_coerce = MLJ.table(input_data[train_indexes, :])
        X_train = coerce(X_train_no_coerce,  autotype(X_train_no_coerce, rules = (:discrete_to_continuous,)))
        y_train = categorical(output_data[train_indexes])
        
        X_test_no_coerce = MLJ.table(input_data[test_indexes, :])
        X_test = coerce(X_test_no_coerce, autotype(X_test_no_coerce, rules = (:discrete_to_continuous,)))
        y_test = categorical(output_data[test_indexes])

        # Extract hyperparameters for each estimator
        base_models_NamedTuple = (; (
            Symbol(estimators[n]) => (
                estimators[n] == :SVC ?
                    getSVCProbabilisticModel(get(modelsHyperParameters, estimators[n], Dict())) :
                    getModel(estimators[n], get(modelsHyperParameters, estimators[n], Dict()))
            )
            for n in 1:length(estimators)
        )...)
        lr = getModel(:LinearRegressor, Dict())
        model_bagging = Stack(; 
            metalearner = lr,
            resampling = Holdout(fraction_train=0.8, shuffle=true, rng=rng),
            measures = log_loss,
            base_models_NamedTuple... 
        )


        mach = machine(model_bagging, X_train, y_train) |> MLJ.fit!

        predictions_prob = MLJ.predict(mach, X_test)
        

        #Convert probabilities in labels and convert it in labels
        predicted_labels = mode.(predictions_prob)
        predicted_labels_str = string.(predicted_labels)

        #Calculate the confusion matrix
        metrics = confusionMatrix(predicted_labels_str, string.(y_test))
        total_confusion_matrix .+= Float32.(metrics.confusion_matrix)

   
        println("metrics for fold $k: $metrics")

        fold_results = Dict(
            :accuracy => metrics.accuracy,
            :error_rate => metrics.error_rate,
            :sensitivity => metrics.sensitivity,
            :specificity => metrics.specificity,
            :ppv => metrics.positive_predictive_value,
            :npv => metrics.negative_predictive_value,
            :f1_score => metrics.f_score
        ) 

        results_vector[k] = fold_results


    end

    #Define dict to return metric (mean and standard desviation) 
    aggregated_results = Dict{Symbol, Tuple{Float64, Float64}}()

    #Iterate over the metric of first fold to get the names (all folds have the same metrics)
    for metric_name in keys(results_vector[1])
        metric_values = [fold[metric_name] for fold in results_vector]
        
        mean_val = mean(metric_values)
        std_val = std(metric_values)
        
        aggregated_results[metric_name] = (mean_val, std_val)
        
        println("$(rpad(metric_name, 12)): Mean value: $(round(mean_val, digits=3)) - STD: $(round(std_val, digits=3))")
    end

    return (
        aggregated_results[:accuracy],
        aggregated_results[:error_rate],
        aggregated_results[:sensitivity],
        aggregated_results[:specificity],
        aggregated_results[:ppv],
        aggregated_results[:npv],
        aggregated_results[:f1_score],
        total_confusion_matrix
    )
end
