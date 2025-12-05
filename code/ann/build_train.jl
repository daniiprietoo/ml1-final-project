using Flux;
using Flux.Losses
using Statistics;
include("../general/utils_general.jl")
include("../general/train_metrics.jl")


"""
    buildClassANN(numInputs, topology, numOutputs; transferFunctions)

Construct a feed-forward neural network for classification using Flux.

The network consists of a sequence of dense hidden layers defined by
`topology` and a final output layer adapted to binary or multiclass
classification.
"""
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    
    return ann;
end; 



#Original train function
"""
    trainClassANNOriginal(topology, dataset; transferFunctions, maxEpochs, minLoss, learningRate)

Train a classification ANN on the full dataset without early stopping or
validation splitting.

Returns the trained network and the history of training losses.
"""
function trainClassANNOriginal(topology::AbstractArray{<:Int,1},      
                    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    (inputs, targets) = dataset;
    
    # This function assumes that each sumple is in a row
    # we are going to check the numeber of samples to have same inputs and targets
    @assert(size(inputs,1)==size(targets,1));

    # We define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Setting up the loss funtion to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # This vectos is going to contain the losses and precission on each training epoch
    trainingLosses = Float32[];

    # Inicialize the counter to 0
    numEpoch = 0;
    # Calcualte the loss without training
    trainingLoss = loss(ann, inputs', targets');
    #  Store this one for checking the evolution.
    push!(trainingLosses, trainingLoss);
    #  and give some feedback on the screen
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Start the training until it reaches one of the stop critteria
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;
        # calculate the loss for this epoch
        trainingLoss = loss(ann, inputs', targets');
        # store it
        push!(trainingLosses, trainingLoss);
        # shown it
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # return the network and the evolution of the error
    return (ann, trainingLosses);      
end

"""
    trainClassANN(topology, trainingDataset; validationDataset, testDataset, ...)

Train a classification ANN with optional validation and test sets and early
stopping based on validation loss.

Returns the best-performing network (according to validation loss) and the
history of training/validation/test losses.
"""
function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
    maxEpochsVal::Int=20, showText::Bool=false) 


    (trainInputs, trainTargets) = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;

    @assert(size(trainInputs,1)==size(trainTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));

    #Define the ANN
    ann = buildClassANN(size(trainInputs,2), topology, size(trainTargets,2));

    # Setting up the loss funtion to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    bestAnn = deepcopy(ann);


    # This vectors is going to contain the losses and precission on each training epoch
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];


    epochsWithoutImprovement=0;
    bestValidationLoss = Inf;

    # Inicialize the counter to 0
    numEpoch = 0;

    if isempty(validationInputs)
        (ann, trainingLosses) = trainClassANNOriginal(topology, (trainInputs, trainTargets); maxEpochs=maxEpochs, learningRate=learningRate);
        return (ann, trainingLosses, validationLosses, testLosses);

    else

        trainingLoss = loss(ann, trainInputs', trainTargets');
        push!(trainingLosses, trainingLoss);

        validationLoss = loss(ann, validationInputs', validationTargets');
        push!(validationLosses, validationLoss);

        testLoss = loss(ann, testInputs', testTargets');
        push!(testLosses, testLoss);

        # Some feedback on the screen
        println("Epoch ", numEpoch, ": -- Training loss ", trainingLoss, " -- Validation loss:", validationLoss, " -- Test loss ", testLoss);


        opt_state = Flux.setup(Adam(learningRate), ann);


        # Start the training until it reaches one of the stop critteria
        while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (epochsWithoutImprovement < maxEpochsVal)

            # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
            Flux.train!(loss, ann, [(trainInputs', trainTargets')], opt_state);

            numEpoch += 1;
            
            trainingLoss = loss(ann, trainInputs', trainTargets');
            push!(trainingLosses, trainingLoss);
            outputsTraining = ann(trainInputs')';
            trainingAccuracy = accuracy(outputsTraining, trainTargets);


            validationLoss = loss(ann, validationInputs', validationTargets');
            push!(validationLosses, validationLoss);
            outputsValidation = ann(validationInputs')';
            validationAccuracy = accuracy(outputsValidation, validationTargets);

            testLoss = loss(ann, testInputs', testTargets');
            push!(testLosses, testLoss);
            outputsTest = ann(testInputs')';
            testAccuracy = accuracy(outputsTest, testTargets);

            #  and give some feedback on the screen
            if numEpoch % 100 == 0
                println("Epoch ", numEpoch, ": -- Training loss ", trainingLoss, " -- Validation loss:", validationLoss, " -- Test loss ", testLoss);
                println("Epoch ", numEpoch, ": -- Training acc  ", trainingAccuracy, " -- Validation acc: ", validationAccuracy, " -- Test acc  ", testAccuracy);
            end

            
            if validationLoss < bestValidationLoss
                bestValidationLoss = validationLoss
                epochsWithoutImprovement = 0
                bestAnn = deepcopy(ann) 
            else
                epochsWithoutImprovement += 1
            end

                  
            if (trainingLoss <= minLoss)
                println("Min loss achieved .")
                break;
            end
            
            if (epochsWithoutImprovement >= maxEpochsVal)
                println("Early stopped: Validation loss hasn't improved in  ", maxEpochsVal, " epochs.")
                break;
            end

        end;

    end;


    return (bestAnn, trainingLosses, validationLosses, testLosses);

end;



"""
    ANNCrossValidation(topology, dataset, crossValidationIndices; ...)

Perform repeated k-fold cross-validation for a classification ANN.

For each fold and for each execution, the network is trained (optionally with a
validation split) and evaluated on the test fold. Metrics and confusion
matrices are averaged across executions and folds.
"""
function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int64=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)
  
  inputs = dataset[1]
  outputs_original = dataset[2]
  
  totalDatasetSize = size(inputs, 1)
  
  
  # One hot encoding
  classes = unique(outputs_original)
  numClasses = length(classes)
  outputs = oneHotEncoding(outputs_original, classes)
  
  
  numFolds = maximum(crossValidationIndices);
  
  accuracy = []
  error_rate = []
  sensitivity = []
  specificity = []
  ppv = []
  npv = []
  f1 = []
  confusion_matrix = zeros(Float32, numClasses, numClasses)
  
  
    for k in 1:numFolds
  
      println("Fold ", k," of ", numFolds)
  
      #Initialize confusion matrix and metrics vectors
      all_confusion_matrices = zeros(Float32, numClasses, numClasses, numExecutions)
  
      accuracy_fold_k = []
      error_rate_fold_k = []
      sensitivity_fold_k = []
      specificity_fold_k = []
      ppv_fold_k = []
      npv_fold_k = []
      f1_fold_k = []
  
  
      test_indexes = (crossValidationIndices .== k)
      train_indexes = .!test_indexes  
  
  
      train_inputs = inputs[train_indexes, :]
      train_outputs = outputs[train_indexes, :]
  
  
      test_inputs = inputs[test_indexes, :]
      test_outputs = outputs[test_indexes, :]
  
      trainDatasetLength = size(train_outputs, 1)
  
  
      for exec in 1:numExecutions
  
  
        if validationRatio > 0
  
          adaptedValidationRatio = (totalDatasetSize * validationRatio) / trainDatasetLength
          (trainIndexes, validationIndexes) = holdOut(trainDatasetLength, adaptedValidationRatio)
          (train_inputs_temp, train_outputs_temp, validation_inputs_temp, validation_outputs_temp) = (train_inputs[trainIndexes, :],  train_outputs[trainIndexes, :] , train_inputs[validationIndexes, :],
          train_outputs[validationIndexes, :] )
  
          (ann, trainingLosses, validationLosses, testLosses) = trainClassANN(topology,
          (train_inputs_temp, train_outputs_temp); validationDataset=(validation_inputs_temp,validation_outputs_temp),
          testDataset=(test_inputs, test_outputs),
          maxEpochs=maxEpochs, 
          learningRate=learningRate, maxEpochsVal=maxEpochsVal, minLoss=minLoss, transferFunctions=transferFunctions);
    
        else
  
          (ann, trainingLosses, validationLosses, testLosses) = trainClassANN(topology,
          (train_inputs, train_outputs);
          maxEpochs=maxEpochs, 
          learningRate=learningRate, maxEpochsVal=maxEpochsVal, minLoss=minLoss, transferFunctions=transferFunctions);
  
        end
  
       
        test_outputs_predicted = Matrix(ann(test_inputs')')
  
        (
          accuracy_exection_result, 
          error_rate_exection_result, 
          sensitivity_exection_result, 
          specificity_exection_result, 
          positivePredictiveValue_exection_result, 
          negativePredictiveValue_exection_result, 
          fScore_exection_result, 
          confusion_matrix_exection_result
        ) = confusionMatrix(test_outputs_predicted, test_outputs)
  
        all_confusion_matrices[:, :, exec] = confusion_matrix_exection_result
        push!(accuracy_fold_k,  accuracy_exection_result)
        push!(error_rate_fold_k,  error_rate_exection_result)
        push!(sensitivity_fold_k, sensitivity_exection_result)
        push!(specificity_fold_k, specificity_exection_result)
        push!(ppv_fold_k,  positivePredictiveValue_exection_result)
        push!(npv_fold_k,  negativePredictiveValue_exection_result)
        push!(f1_fold_k,   fScore_exection_result)
  
  
      end
  
      push!(accuracy, mean(accuracy_fold_k))
      push!(error_rate, mean(error_rate_fold_k))
      push!(sensitivity, mean(sensitivity_fold_k))
      push!(specificity, mean(specificity_fold_k))
      push!(ppv, mean(ppv_fold_k))
      push!(npv, mean(npv_fold_k))
      push!(f1, mean(f1_fold_k))
  
      # Accumulate confusion matrix for this fold
      fold_confusion_matrix = dropdims(mean(all_confusion_matrices, dims=3), dims=3)
      confusion_matrix .+= fold_confusion_matrix
  
    end
  
  
    results = (
      (mean(accuracy),    std(accuracy)),
      (mean(error_rate),  std(error_rate)),
      (mean(sensitivity), std(sensitivity)),
      (mean(specificity), std(specificity)),
      (mean(ppv),         std(ppv)),
      (mean(npv),         std(npv)),
      (mean(f1),     std(f1)),
      confusion_matrix
  )
  
  return results
  
  
  
  end;
