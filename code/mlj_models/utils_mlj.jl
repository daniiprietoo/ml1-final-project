"""
    prepare_data_for_mlj(input_data::Matrix, output_data::CategoricalArray, train_split_ratio::Float64)

Prepare input and output data for training and evaluating MLJ models.

The function:
- Normalizes the input features to the [0, 1] range using `normalizeMinMax!`.
- Splits the dataset into train and test subsets according to `train_split_ratio` using a hold-out split.
- Wraps the input matrices as MLJ tables.
- Automatically infers and coerces feature types (e.g. discrete to continuous).
- Ensures the target variables are categorical.

# Arguments
- `input_data`: Feature matrix (observations in rows, features in columns).
- `output_data`: Categorical target vector (labels for each observation).
- `train_split_ratio`: Fraction of data to use for training (e.g. 0.7).

# Returns
A tuple `(train_input, train_output, test_input, test_output)` where:
- `train_input`: MLJ table of training features.
- `train_output`: Categorical array of training labels.
- `test_input`: MLJ table of test features.
- `test_output`: Categorical array of test labels.
"""
function prepare_data_for_mlj(input_data::Matrix, output_data::CategoricalArray, train_split_ratio::Float64)
    
    input_data = normalizeMinMax!(convert(Array{Float32,2}, input_data[:,1:end]))
    datasetLength = size(input_data, 1)
    (trainIndexes, testIndexes) = holdOut(datasetLength, train_split_ratio)

    train_input_no_coerce, train_output_no_coerce = (MLJ.table(input_data[trainIndexes, :]), output_data[trainIndexes])
    test_input_no_coerce, test_output_no_coerce = (MLJ.table(input_data[testIndexes, :]), output_data[testIndexes])
  
   train_input, train_output = (coerce(train_input_no_coerce, autotype(train_input_no_coerce, rules=(:discrete_to_continuous,))), categorical(train_output_no_coerce))
   test_input, test_output = (coerce(test_input_no_coerce, autotype(test_input_no_coerce, rules=(:discrete_to_continuous,))), categorical(test_output_no_coerce))

    return (train_input, train_output, test_input, test_output)
end
