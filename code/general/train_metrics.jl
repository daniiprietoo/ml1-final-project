using Statistics, LinearAlgebra
include("utils_general.jl")

# PRINT METRICS

"""
    printMetrics(metrics)

Pretty-print the main classification metrics and confusion matrix.

The input is expected to be a named tuple as returned by `confusionMatrix`.
"""
function printMetrics(metrics)
   accuracy, sensitivity, specificity, positive_predictive, negative_predictive_value, f_score, confusion_matrix = metrics
    println("Accuracy: ", metrics.accuracy)
    println("Error rate: ", metrics.error_rate)
    println("Sensitivity: ", metrics.sensitivity)
    println("Specificty: ", metrics.specificity)
    println("Positive Predictive: ", metrics.positive_predictive_value)
    println("Negative Predictive: ", metrics.negative_predictive_value)
    println("F-score: ", metrics.f_score)
    println("Confusion matrix: ", metrics.confusion_matrix)
end


# CONFUSION MATRIX

"""
    confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Compute binary classification metrics and confusion matrix from Boolean
predictions and targets.
"""
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert length(outputs) == length(targets) "Outputs and targets must have the same length."

    tp = sum(outputs .& targets)
    tn = sum((.!outputs) .& (.!targets))
    fp = sum(.!outputs .& (targets))
    fn = sum((outputs) .& (.!targets))

    total_patterns = length(targets)
    
    
    if tn == total_patterns
        sensitivity = 1.0
        positive_predictive_value = 1.0
    else
        sensitivity = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        positive_predictive_value = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    end
    
    if tp == total_patterns
        specificity = 1.0
        negative_predictive_value = 1.0
    else
        specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0.0
        negative_predictive_value = (tn + fn) > 0 ? tn / (tn + fn) : 0.0
    end

    accuracy = total_patterns > 0 ? (tp + tn) / total_patterns : 0.0
    error_rate = total_patterns > 0 ? (fp + fn) / total_patterns : 0.0
    
    f_score = (positive_predictive_value + sensitivity) > 0 ? 
              2 * (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity) : 0.0

    confusion_matrix = [tn fp; fn tp]
    
    return (
        accuracy=accuracy, 
        error_rate=error_rate, 
        sensitivity=sensitivity, 
        specificity=specificity, 
        positive_predictive_value=positive_predictive_value, 
        negative_predictive_value=negative_predictive_value, 
        f_score=f_score, 
        confusion_matrix=confusion_matrix
    )
end


"""
    confusionMatrix(outputs::AbstractArray{<:Real,1}, targets; threshold=0.5)

Compute binary confusion matrix and metrics from real-valued scores and Boolean
targets by thresholding the scores.
"""
function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    realToBoolean(outputs, threshold) = outputs.>threshold
    outputsBoolean = confusionMatrix(realToBoolean(outputs, threshold), targets)
end


"""
    confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; mode=:macro)

Compute multiclass confusion matrix and aggregate metrics for one-hot encoded
predictions and targets.

`mode` controls macro-averaged (unweighted) or support-weighted averaging of
per-class metrics.
"""
function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2}; mode::Symbol=:macro)

    num_columns_out = size(outputs, 2)
    num_columns_target = size(targets, 2)

    num_rows_out = size(outputs, 1)
    num_rows_target = size(targets, 1)

    @assert num_columns_out != 2 "The number of colums cannot be 2."
    @assert num_columns_out == num_columns_target "Input matrices must have the same number of columns"
    @assert num_rows_out == num_rows_target "Input matrices must have the same number of rows"

    @assert mode === :macro || mode === :weighted "Mode must be either :macro or :weighted."


    if num_columns_out === 1
        return confusionMatrix(vec(outputs), vec(targets))
    end

    num_classes = num_columns_out

    sensitivities = zeros(Float64, num_classes)
    specificities = zeros(Float64, num_classes)
    ppvs = zeros(Float64, num_classes) 
    npvs = zeros(Float64, num_classes)
    f_scores = zeros(Float64, num_classes)

    # Iterate over each class to calculate  metrics
    for i in 1:num_classes
        # Treat each class as a separate binary problem
        class_outputs = outputs[:, i]
        class_targets = targets[:, i]
        
        if any(class_targets) && any(class_outputs)
            metrics = confusionMatrix(class_outputs, class_targets)
            sensitivities[i] = metrics.sensitivity
            specificities[i] = metrics.specificity
            ppvs[i] = metrics.positive_predictive_value
            npvs[i] = metrics.negative_predictive_value
            f_scores[i] = metrics.f_score
        end
    end


    #Version with comprehension
    conf_matrix_multiclass = [sum(targets[:, i] .& outputs[:, j]) for i in 1:num_classes, j in 1:num_classes]

    
    if mode === :macro
        sensitivity = mean(sensitivities) 
        specificity = mean(specificities)
        ppv = mean(ppvs)
        npv = mean(npvs)
        f_score = mean(f_scores)
    else mode === :weighted
        support = vec(sum(targets, dims=1))
        total_patterns = sum(support)
        sensitivity = dot(sensitivities, support) / total_patterns
        specificity = dot(specificities, support) / total_patterns
        ppv = dot(ppvs, support) / total_patterns
        npv = dot(npvs, support) / total_patterns
        f_score = dot(f_scores, support) / total_patterns
    end

   accuracy_result = accuracy(outputs, targets)

    return (
        accuracy=accuracy_result,
        error_rate = 1-accuracy_result,
        sensitivity=sensitivity,
        specificity=specificity,
        positive_predictive_value=ppv,
        negative_predictive_value=npv,
        f_score=f_score,
        confusion_matrix=conf_matrix_multiclass
    )
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold=0.5, weighted=true)

Compute multiclass confusion matrix and metrics from real-valued score
matrices, thresholding them to Booleans before evaluation.
"""
function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    realToBoolean(outputs::AbstractArray{<:Real,2}, threshold::Real) = outputs.>threshold
    mode = weighted ? :weighted : :macro
    outputsBoolean = confusionMatrix(realToBoolean(outputs, threshold), targets; mode=mode)
end


"""
    confusionMatrix(outputs, targets, classes; weighted=true)

Compute multiclass confusion matrix and metrics for arbitrary labels, given an
explicit list of classes.
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)

    @assert all([in(target, unique(classes)) for target in targets]) "Not all labels in targets are present in classes.";
    @assert all([in(output, unique(classes)) for output in outputs]) "Not all labels in outputs are present in classes.";

    (outputsOneHotEncoding, targetsOneHotEncoding) = (oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes))

    mode = weighted ? :weighted : :macro

    confusionMatrix(outputsOneHotEncoding, targetsOneHotEncoding, mode=mode)

end;


"""
    confusionMatrix(outputs, targets; weighted=true)

Compute multiclass confusion matrix and metrics for arbitrary labels, inferring
the set of classes from `outputs` and `targets`.
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    confusionMatrix(outputs, targets, classes, weighted=weighted)
end


## PRINT CONFUSION MATRIX

"""
    printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Compute and print binary confusion matrix and metrics for Boolean outputs and
targets.
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
        metrics = confusionMatrix(outputs, targets)
        printMetrics(metrics)
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold=0.5)

Threshold real-valued scores and print the resulting binary confusion matrix
and metrics.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    metrics = confusionMatrix(outputs, targets)
    printMetrics(metrics)
end


"""
    printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted=true)

Print multiclass confusion matrix and metrics for one-hot encoded Boolean
outputs and targets.
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

    mode = weighted ? :weighted : :macro

    metrics = confusionMatrix(outputs, targets, mode=mode)
    printMetrics(metrics)
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted=true)

Threshold real-valued score matrices and print the resulting multiclass
confusion matrix and metrics.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    mode = weighted ? :weighted : :macro
    metrics = confusionMatrix(outputs, targets, weighted=weighted)
    printMetrics(metrics)
end

"""
    printConfusionMatrix(outputs, targets, classes; weighted=true)

Print confusion matrix and metrics for arbitrary labels and an explicit list of
classes.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets, classes, weighted=weighted)
    printMetrics(metrics)
end

"""
    printConfusionMatrix(outputs, targets; weighted=true)

Print confusion matrix and metrics for arbitrary labels, inferring the set of
classes automatically.
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets, weighted=weighted)
    printMetrics(metrics)
end
