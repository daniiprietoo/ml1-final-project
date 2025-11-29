# ...existing code...

using CSV
using DataFrames

# Utils function to load and merge the data
function load_and_merge_data(tracks_file::String, features_file::String; 
                             selected_tracks_columns::Union{Nothing, Vector{Symbol}} = nothing,
                             selected_features_algorithms::Union{Nothing, Vector{Symbol}} = nothing) ::DataFrame
    tracks_df = load_tracks(tracks_file, selected_tracks_columns)
    features_df = load_features(features_file, selected_features_algorithms)
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
        selected_indices = 1:length(rows[2])
    else
        # Find indices where rows[1] is in selected_algorithms
        selected_indices = findall(x -> Symbol(x) in selected_algorithms, rows[1])
    end
    
    # Load features_df with header=false
    features_df = CSV.read(features_file, DataFrame; header=false, skipto=5, select=[1; selected_indices .+ 1])
    
    # Create combined column names
    combined_names = [:track_id]
    for i in selected_indices
        push!(combined_names, Symbol(string(rows[1][i]) * "_" * string(rows[2][i])))
    end
    
    rename!(features_df, combined_names, makeunique=true)
    
    return features_df
end

# Example usage:
# To load all:
# df_final = load_and_merge_data(TRACKS_FILE, FEATURES_FILE)

# To load specific tracks columns and features algorithms:
# df_final = load_and_merge_data(TRACKS_FILE, FEATURES_FILE; selected_tracks_columns=[:listens, :duration], selected_features_algorithms=[:mfcc, :chroma])

