include("./models.jl")
using Statistics
using MLJ


function mljCrossValidation(
    modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Vector{Int})

    inputs, targets_original = dataset
    
    targets = string.(targets_original)
    classes = unique(targets)
    numClasses = length(classes)
    numFolds = maximum(crossValidationIndices)

    accuracies = zeros(Float64, numFolds)
    error_rates = zeros(Float64, numFolds)
    sensitivities = zeros(Float64, numFolds)
    specificities = zeros(Float64, numFolds)
    ppvs = zeros(Float64, numFolds)
    npvs = zeros(Float64, numFolds)
    f1_scores = zeros(Float64, numFolds)
    total_confusion_matrix = zeros(Float64, numClasses, numClasses)



    # Bucle de validación cruzada
    for k in 1:numFolds
        println("fold ", k, " de ", numFolds, " para el modelo ", modelType)

        test_indexes = (crossValidationIndices .== k)
        train_indexes = .!test_indexes
        
        train_inputs = inputs[train_indexes, :]
        train_targets = targets[train_indexes]
        
        test_inputs = inputs[test_indexes, :]
        test_targets = targets[test_indexes]   

        model = nothing
        if modelType == :SVC
            model = getSVCModel(modelHyperparameters)
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
        else
            error("Tipo de modelo no soportado: ", modelType)
        end

        mach = machine(model, MLJ.table(train_inputs), categorical(train_targets))
        MLJ.fit!(mach, verbosity=0)
        
        predictions_prob = MLJ.predict(mach, MLJ.table(test_inputs))
        
        if modelType == :SVC
            predicted_labels = predictions_prob
        else
            predicted_labels = mode.(predictions_prob)
        end
        
        predicted_labels_str = string.(predicted_labels)

        metrics = confusionMatrix(predicted_labels_str, test_targets)
        
        accuracies[k] = metrics.accuracy
        error_rates[k] = metrics.error_rate
        sensitivities[k] = metrics.sensitivity
        specificities[k] = metrics.specificity
        ppvs[k] = metrics.positive_predictive_value
        npvs[k] = metrics.negative_predictive_value
        f1_scores[k] = metrics.f_score
        
        total_confusion_matrix .+= metrics.confusion_matrix
    end

    return (
       (mean(accuracies), std(accuracies)),
       (mean(error_rates), std(error_rates)),
       (mean(sensitivities),std(sensitivities)),
       (mean(specificities), std(specificities)),
       (mean(ppvs),std(ppvs)),
       (mean(npvs),std(npvs)),
       (mean(f1_scores),std(f1_scores)),
        total_confusion_matrix
    )
end
