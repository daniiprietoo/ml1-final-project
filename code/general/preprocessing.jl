# Utils function to load and merge the data
using CSV
using DataFrames
using MultivariateStats
using Statistics
using LinearAlgebra
using MLJ

PCA = @load PCA pkg=MultivariateStats
LDA = @load LDA pkg=MultivariateStats

function load_and_merge_data(tracks_file::String, features_file::String; 
                             selected_tracks_columns::Union{Nothing, Vector{Symbol}} = nothing,
                             selected_features_algorithms::Union{Nothing, Vector{Symbol}} = nothing) ::DataFrame
    tracks_df = load_tracks(tracks_file, selected_columns=selected_tracks_columns)
    features_df = load_features(features_file; selected_algorithms=selected_features_algorithms)
    # Merge on track_id
    tracks_df.track_id = string.(tracks_df.track_id)
    features_df.track_id = string.(features_df.track_id)
    df_final = innerjoin(tracks_df, features_df, on=:track_id)
    
    return df_final
end

function load_tracks(tracks_file::String; selected_columns::Union{Nothing, Vector{Symbol}} = nothing) ::DataFrame
    @assert isfile(tracks_file) "$tracks_file not found, please specify the relative directory to this file"
    # Read headers
    header_preview = CSV.File(tracks_file; header=false, limit=2)
    rows = [collect(row) for row in header_preview]
    
    track_indices = findall(x -> string(x) == "track", rows[1])
    
    if selected_columns === nothing
        # Filter track_indices to valid column indices
        selected_indices = filter(i -> i + 1 <= length(rows[2]) + 1, track_indices)  # Assuming DataFrame has length(rows[2]) + 1 columns
        
        # Load with track columns only
        tracks_df = CSV.read(tracks_file, DataFrame; header=false, skipto=4, select=[1; selected_indices])
        
        # Create combined column names
        combined_names = [:track_id]
        for i in selected_indices
            push!(combined_names, Symbol("track_" * string(rows[2][i])))
        end
        
        rename!(tracks_df, combined_names)
    else
        # Find indices of selected columns within track group only
        selected_indices = Int[]
        for col in selected_columns
            idx = findfirst(i -> string(rows[2][i]) == string(col) && string(rows[1][i]) == "track", 1:length(rows[2]))
            if idx !== nothing && idx + 1 <= length(rows[2]) + 1
                push!(selected_indices, idx)
            else
                @warn "Column '$col' not found in track columns or index out of range, skipping."
            end
        end
        
        # Load tracks_df with header=false, selecting only the correct indices
        tracks_df = CSV.read(tracks_file, DataFrame; header=false, skipto=4, select=[1; selected_indices])
        
        # Create combined column names
        combined_names = [:track_id]
        for i in selected_indices
            push!(combined_names, Symbol("track_" * string(rows[2][i])))
        end
        
        rename!(tracks_df, combined_names)
    end
    
    return tracks_df
end

function load_features(features_file::String; selected_algorithms::Union{Nothing, Vector{Symbol}} = nothing) ::DataFrame
    @assert isfile(features_file) "$features_file not found, please specify the relative directory to this file"
    # Read headers
    header_preview = CSV.File(features_file; header=false, limit=2)
    rows = [collect(row) for row in header_preview]
    
    if selected_algorithms === nothing
        selected_indices = 2:length(rows[2])
    else
        # Find indices where rows[1] is in selected_algorithms
        selected_indices = findall(x -> Symbol(x) in selected_algorithms, rows[1])
    end
    
    # Convert selected_indices to a vector to ensure proper handling
    selected_indices_vec = collect(selected_indices)
    
    # Select columns: column 1 (track_id) + feature columns
    selected_cols = [1; selected_indices_vec]
    features_df = CSV.read(features_file, DataFrame; header=false, skipto=5, select=selected_cols)
    
    # Create combined column names based on actual columns read
    # First column is always track_id
    combined_names = [:track_id]
    
    # Add names for feature columns
    num_feature_cols = ncol(features_df) - 1
    
    # Only create names for the actual number of feature columns we have
    for i in 1:min(length(selected_indices_vec), num_feature_cols)
        idx = selected_indices_vec[i]
        if idx <= length(rows[1]) && idx <= length(rows[2])
            push!(combined_names, Symbol(string(rows[1][idx]) * "_" * string(rows[2][idx])))
        else
            # Fallback if index is out of bounds
            push!(combined_names, Symbol("feature_$i"))
        end
    end
    
    # If we still don't have enough names, add generic ones
    while length(combined_names) < ncol(features_df)
        push!(combined_names, Symbol("feature_$(length(combined_names))"))
    end
    
    rename!(features_df, combined_names, makeunique=true)
    
    return features_df
end

# Example usage:
# To load all:
# df_final = load_and_merge_data(TRACKS_FILE, FEATURES_FILE)

# To load specific tracks columns and features algorithms:
# df_final = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens, :duration], selected_features_algorithms=[:mfcc, :chroma])

function getPCAModel(modelHyperparameters::Dict) 
    if haskey(modelHyperparameters, :variance_ratio) && !isempty(modelHyperparameters[:variance_ratio])
        variance_ratio = get(modelHyperparameters, :variance_ratio, 0.95)
        return PCA(variance_ratio=variance_ratio)
    elseif haskey(modelHyperparameters, :maxoutdim) && !isempty(modelHyperparameters[:maxoutdim])
        maxoutdim = get(modelHyperparameters, :maxoutdim, 25)
        return PCA(maxoutdim=maxoutdim)
    else
        error("PCA accepts either `variance_ratio` or `maxoutdim`")
    end
        
end

function getLDAModel(modelHyperparameters::Dict) 
    if haskey(modelHyperparameters, :outdim) && !isempty(modelHyperparameters[:outdim])
        outdim = get(modelHyperparameters, :outdim, 2)
        return LDA(outdim=outdim)
    end
    
    return LDA()  
end

function apply_pca_mlj(data::AbstractArray{<:Real,2}; maxoutdim::Union{Int, Nothing}=nothing, variance_ratio::Union{Float64, Nothing}=nothing)
    # Convert to MLJ table format
    data_table = MLJ.table(data)
    
    # Create PCA model
    if variance_ratio !== nothing
        pca_model = getPCAModel(Dict(:variance_ratio => variance_ratio))
    elseif maxoutdim !== nothing
        pca_model = getPCAModel(Dict(:maxoutdim => maxoutdim))
    else
        error("Either n_components or variance_ratio must be specified")
    end
    
    # Fit and transform
    pca_mach = machine(pca_model, data_table) |> MLJ.fit!
    transformed_table = MLJ.transform(pca_mach, data_table)
    
    # Convert back to matrix
    transformed_data = MLJ.matrix(transformed_table)
    
    return transformed_data, pca_mach
end

function transform_pca_mlj(pca_mach, data::AbstractArray{<:Real,2})
    data_table = MLJ.table(data)
    transformed_table = MLJ.transform(pca_mach, data_table)
    return MLJ.matrix(transformed_table)
end

function apply_lda_mlj(data::AbstractArray{<:Real,2}, labels::Vector{String}; outdim::Union{Int, Nothing}=nothing)
    # Convert to MLJ table format
    data_table = MLJ.table(data)
    labels_categorical = categorical(labels)
    
    # Create LDA model
    if outdim !== nothing
        lda_model = getLDAModel(Dict(:outdim => outdim))
    else 
        lad_model = getLDAModel()
    end
    # Fit and transform
    lda_mach = machine(lda_model, data_table, labels_categorical) |> MLJ.fit!
    transformed_table = MLJ.transform(lda_mach, data_table)
    
    # Convert back to matrix
    transformed_data = MLJ.matrix(transformed_table)
    
    return transformed_data, lda_mach
end

function transform_lda_mlj(lda_mach, data::AbstractArray{<:Real,2})
    data_table = MLJ.table(data)
    transformed_table = MLJ.transform(lda_mach, data_table)
    return MLJ.matrix(transformed_table)
end


function extract_features(df)
    # Extract feature columns (exclude track_id and listens)
    feature_cols = [col for col in names(df) if col != "track_listens" && col != "track_id"]
    features_df = select(df, feature_cols)
    print(feature_cols)
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
    return features_df
end

