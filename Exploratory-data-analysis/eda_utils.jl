### Fixed utility functions for FMA Dataset EDA
### Adapted to work with multi-level CSV column headers

using CSV
using DataFrames
using Statistics
using StatsBase
using Plots
using StatsPlots
using Dates
using LinearAlgebra
using MultivariateStats
using Clustering
using Distributions
using HypothesisTests
using GLM
using Markdown

println("Libraries loaded successfully!")

#%% ============================================================================
# HELPER FUNCTIONS FOR COLUMN NAME MAPPING
# ==============================================================================

"""
Find column by partial name match
"""
function find_column(df::DataFrame, pattern::String)
    matches = filter(x -> occursin(lowercase(pattern), lowercase(x)), names(df))
    return isempty(matches) ? nothing : matches[1]
end

"""
Find all columns matching a pattern
"""
function find_columns(df::DataFrame, pattern::String)
    return filter(x -> occursin(lowercase(pattern), lowercase(x)), names(df))
end

#%% ============================================================================
# SECTION 1: DATA LOADING AND INITIAL INSPECTION
# ==============================================================================

"""
Load and prepare the tracks and features datasets
"""
function load_datasets(tracks_path="data/tracks.csv", 
                       features_path="data/features.csv")
    println("\n=== Loading Datasets ===")
    
    # Load tracks with multi-level column structure
    tracks_raw = CSV.read(tracks_path, DataFrame, header=[1, 2, 3], 
                          skipto=4, missingstring=["", "NA", "NaN"])
    
    # Load features
    features_raw = CSV.read(features_path, DataFrame, 
                            header=[1, 2, 3], skipto=4, 
                            missingstring=["", "NA", "NaN"])
    
    println("Tracks shape: $(size(tracks_raw))")
    println("Features shape: $(size(features_raw))")
    
    return tracks_raw, features_raw
end

"""
Basic data quality assessment
"""
function data_quality_report(df::DataFrame, name::String)
    println("\n" * "="^80)
    println("DATA QUALITY REPORT: $name")
    println("="^80)
    
    # Dimensions
    n_rows, n_cols = size(df)
    println("\nDimensions: $n_rows rows × $n_cols columns")
    
    # Missing values analysis
    println("\nMissing Values Analysis:")
    missing_counts = Dict{String, Int}()
    for col in names(df)
        n_missing = sum(ismissing.(df[!, col]))
        if n_missing > 0
            missing_counts[col] = n_missing
        end
    end
    
    if !isempty(missing_counts)
        sorted_missing = sort(collect(missing_counts), by=x->x[2], rev=true)
        println("\nTop 10 columns with missing values:")
        for (i, (col, count)) in enumerate(sorted_missing[1:min(10, length(sorted_missing))])
            pct = round(100 * count / n_rows, digits=2)
            println("  $i. $col: $count ($pct%)")
        end
    else
        println("  No missing values detected!")
    end
    
    # Data types
    println("\nData Types Distribution:")
    type_counts = countmap([eltype(df[!, col]) for col in names(df)])
    for (dtype, count) in sort(collect(type_counts), by=x->x[2], rev=true)
        println("  $dtype: $count columns")
    end
    
    return missing_counts
end

#%% ============================================================================
# SECTION 2: TRACKS METADATA ANALYSIS
# ==============================================================================

"""
Analyze temporal patterns in track creation and release
"""
function analyze_temporal_patterns(tracks::DataFrame)
    println("\n" * "="^80)
    println("TEMPORAL ANALYSIS")
    println("="^80)
    
    # Extract date columns
    date_cols = find_columns(tracks, "date")
    
    println("\nAvailable date columns:")
    for col in date_cols
        println("  - $col")
    end
    
    # Try to find and analyze track creation date
    date_created_col = find_column(tracks, "track_date_created")
    
    if ! isnothing(date_created_col)
        dates = tracks[!, date_created_col]
        valid_dates = collect(skipmissing(dates))
        
        if !isempty(valid_dates)
            println("\nTrack Creation Date Range:")
            println("  Earliest: $(minimum(valid_dates))")
            println("  Latest: $(maximum(valid_dates))")
            println("  Total tracks with dates: $(length(valid_dates))")
        end
    else
        println("\nNote: Track creation date column not found or has different name")
    end
    
    return nothing
end

"""
Analyze genre distribution and hierarchy
"""
function analyze_genres(tracks::DataFrame)
    println("\n" * "="^80)
    println("GENRE ANALYSIS")
    println("="^80)
    
    # Find genre-related columns
    genre_cols = find_columns(tracks, "genre")
    println("\nGenre columns found: $genre_cols")
    
    # Try to find top-level genre column
    genre_top_col = find_column(tracks, "genre_top")
    
    if !isnothing(genre_top_col)
        genre_top = tracks[!, genre_top_col]
        valid_genres = collect(skipmissing(genre_top))
        
        if ! isempty(valid_genres)
            genre_counts = countmap(valid_genres)
            
            println("\nTop 15 Genres by Track Count:")
            sorted_genres = sort(collect(genre_counts), by=x->x[2], rev=true)
            for (i, (genre, count)) in enumerate(sorted_genres[1:min(15, length(sorted_genres))])
                pct = round(100 * count / length(valid_genres), digits=2)
                println("  $i. $genre: $count tracks ($pct%)")
            end
            
            # Visualization
            top_n = min(15, length(sorted_genres))
            top_genres = sorted_genres[1:top_n]
            genres_names = [string(x[1]) for x in top_genres]
            genres_values = [x[2] for x in top_genres]
            
            p1 = bar(genres_names, genres_values, 
                     title="Top Genres Distribution",
                     xlabel="Genre", ylabel="Number of Tracks",
                     xrotation=45, legend=false, 
                     color=:steelblue, alpha=0.8,
                     size=(1200, 600),
                     bottom_margin=10Plots.mm)
            display(p1)
            savefig(p1, "genre_distribution.png")
            println("\n✓ Saved: genre_distribution.png")
            
            return genre_counts
        end
    else
        println("\nWarning: Could not find genre_top column")
        println("Available genre columns: $genre_cols")
    end
    
    return nothing
end

"""
Analyze track duration statistics
"""
function analyze_duration(tracks::DataFrame)
    println("\n" * "="^80)
    println("DURATION ANALYSIS")
    println("="^80)
    
    # Find duration column
    duration_col = find_column(tracks, "duration")
    
    if !isnothing(duration_col)
        all_durations = tracks[!, duration_col]
        durations = collect(skipmissing(all_durations))
        
        # Filter for numeric values only
        durations = filter(x -> x isa Number && !isnan(x) && x > 0, durations)
        
        if !isempty(durations)
            println("\nDuration Statistics (seconds):")
            println("  Count: $(length(durations))")
            println("  Mean: $(round(mean(durations), digits=2))")
            println("  Median: $(round(median(durations), digits=2))")
            println("  Std Dev: $(round(std(durations), digits=2))")
            println("  Min: $(round(minimum(durations), digits=2))")
            println("  Max: $(round(maximum(durations), digits=2))")
            println("  25th Percentile: $(round(quantile(durations, 0.25), digits=2))")
            println("  75th Percentile: $(round(quantile(durations, 0.75), digits=2))")
            
            # Total dataset duration
            total_hours = sum(durations) / 3600
            total_days = total_hours / 24
            println("\nTotal Duration:")
            println("  Hours: $(round(total_hours, digits=2))")
            println("  Days: $(round(total_days, digits=2))")
            
            # Visualization - filter for reasonable display
            # Fixed: Use proper array indexing with broadcasting
            display_durations = filter(x -> x < 800, durations)
            
            p2 = histogram(display_durations, 
                          bins=50,
                          title="Track Duration Distribution",
                          xlabel="Duration (seconds)",
                          ylabel="Frequency",
                          legend=false,
                          color=:steelblue,
                          alpha=0.7,
                          size=(1000, 600))
            display(p2)
            savefig(p2, "duration_distribution.png")
            println("\n✓ Saved: duration_distribution.png")
            
            # Box plot
            boxplot_durations = filter(x -> x < 1000, durations)
            p3 = boxplot(["Duration"], boxplot_durations,
                        title="Duration Box Plot",
                        ylabel="Duration (seconds)",
                        legend=false,
                        color=:steelblue,
                        size=(600, 600))
            display(p3)
            savefig(p3, "duration_boxplot.png")
            println("✓ Saved: duration_boxplot.png")
        else
            println("\nNo valid duration data found")
        end
    else
        println("\nWarning: Could not find duration column")
    end
    
    return nothing
end

"""
Analyze user engagement metrics (listens, favorites, comments)
"""
function analyze_engagement(tracks::DataFrame)
    println("\n" * "="^80)
    println("USER ENGAGEMENT ANALYSIS")
    println("="^80)
    
    # Search for engagement columns
    engagement_patterns = ["listens", "favorites", "comments", "interest"]
    found_metrics = Dict{String, String}()
    
    for pattern in engagement_patterns
        col = find_column(tracks, pattern)
        if !isnothing(col)
            found_metrics[pattern] = col
        end
    end
    
    if isempty(found_metrics)
        println("\nNo engagement metrics found")
        return nothing
    end
    
    println("\nFound engagement metrics:")
    for (metric, col) in found_metrics
        println("  - $metric: $col")
    end
    
    # Analyze each metric
    plots_list = []
    for (metric, col) in found_metrics
        all_values = tracks[!, col]
        values = collect(skipmissing(all_values))
        values = filter(x -> x isa Number && !isnan(x) && x >= 0, values)
        
        if !isempty(values)
            println("\n$metric Statistics:")
            println("  Count: $(length(values))")
            println("  Mean: $(round(mean(values), digits=2))")
            println("  Median: $(round(median(values), digits=2))")
            println("  Max: $(maximum(values))")
            if length(values) > 1
                println("  90th Percentile: $(round(quantile(values, 0.90), digits=2))")
            end
            
            # Filter outliers for visualization
            if length(values) > 1
                threshold = quantile(values, 0.95)
                filtered_values = filter(x -> x <= threshold, values)
                
                if !isempty(filtered_values)
                    p = histogram(filtered_values,
                                title=uppercasefirst(replace(metric, "_" => " ")),
                                xlabel="Count",
                                ylabel="Frequency",
                                legend=false,
                                color=:steelblue,
                                alpha=0.7,
                                bins=min(50, length(unique(filtered_values))))
                    push!(plots_list, p)
                end
            end
        end
    end
    
    # Create combined plot
    if !isempty(plots_list)
        n_plots = length(plots_list)
        layout_rows = Int(ceil(n_plots / 2))
        layout_cols = min(2, n_plots)
        
        p_combined = plot(plots_list..., 
                         layout=(layout_rows, layout_cols), 
                         size=(1400, layout_rows * 500))
        display(p_combined)
        savefig(p_combined, "engagement_metrics.png")
        println("\n✓ Saved: engagement_metrics. png")
    end
    
    return nothing
end

#%% ============================================================================
# SECTION 3: AUDIO FEATURES ANALYSIS
# ==============================================================================

"""
Extract and organize feature columns by type
"""
function organize_features(features::DataFrame)
    println("\n" * "="^80)
    println("FEATURE ORGANIZATION")
    println("="^80)
    
    feature_groups = Dict{String, Vector{String}}()
    
    # Common feature types in audio analysis
    feature_types = ["mfcc", "chroma", "spectral", "zcr", "rmse", 
                     "tonnetz", "contrast", "bandwidth", "centroid", "rolloff"]
    
    for ftype in feature_types
        matching_cols = find_columns(features, ftype)
        if !isempty(matching_cols)
            feature_groups[ftype] = matching_cols
            println("  $ftype: $(length(matching_cols)) features")
        end
    end
    
    return feature_groups
end

"""
Compute comprehensive statistics for feature groups
"""
function compute_feature_statistics(features::DataFrame, feature_groups::Dict)
    println("\n" * "="^80)
    println("FEATURE STATISTICS")
    println("="^80)
    
    stats_dict = Dict{String, DataFrame}()
    
    for (group_name, cols) in feature_groups
        if !isempty(cols)
            # Extract numeric columns only
            numeric_cols = String[]
            for col in cols
                col_type = eltype(features[!, col])
                if col_type <: Union{Missing, Number} || col_type <: Number
                    push!(numeric_cols, col)
                end
            end
            
            if !isempty(numeric_cols)
                # Compute statistics
                means = Float64[]
                stds = Float64[]
                mins = Float64[]
                maxs = Float64[]
                medians = Float64[]
                skews = Float64[]
                kurts = Float64[]
                
                for col in numeric_cols
                    values = collect(skipmissing(features[!, col]))
                    values = filter(x -> x isa Number && !isnan(x), values)
                    
                    if !isempty(values)
                        push!(means, mean(values))
                        push!(stds, std(values))
                        push!(mins, minimum(values))
                        push!(maxs, maximum(values))
                        push!(medians, median(values))
                        push!(skews, skewness(values))
                        push!(kurts, kurtosis(values))
                    else
                        push!(means, NaN)
                        push!(stds, NaN)
                        push!(mins, NaN)
                        push!(maxs, NaN)
                        push!(medians, NaN)
                        push!(skews, NaN)
                        push!(kurts, NaN)
                    end
                end
                
                group_stats = DataFrame(
                    feature = numeric_cols,
                    mean = means,
                    std = stds,
                    min = mins,
                    max = maxs,
                    median = medians,
                    skewness = skews,
                    kurtosis = kurts
                )
                
                stats_dict[group_name] = group_stats
                
                valid_means = filter(! isnan, means)
                valid_stds = filter(!isnan, stds)
                
                if !isempty(valid_means) && !isempty(valid_stds)
                    println("\n$group_name Features:")
                    println("  Count: $(length(numeric_cols))")
                    println("  Mean range: [$(round(minimum(valid_means), digits=4)), $(round(maximum(valid_means), digits=4))]")
                    println("  Std range: [$(round(minimum(valid_stds), digits=4)), $(round(maximum(valid_stds), digits=4))]")
                end
            end
        end
    end
    
    return stats_dict
end

"""
Visualize feature distributions
"""
function visualize_feature_distributions(features::DataFrame, feature_groups::Dict)
    println("\n" * "="^80)
    println("FEATURE DISTRIBUTION VISUALIZATION")
    println("="^80)
    
    # MFCC Analysis
    if "mfcc" in keys(feature_groups)
        mfcc_cols = filter(x -> occursin("mean", lowercase(x)), feature_groups["mfcc"])
        
        if length(mfcc_cols) >= 12
            selected_mfccs = mfcc_cols[1:min(12, length(mfcc_cols))]
            
            plots_list = []
            for (i, col) in enumerate(selected_mfccs)
                values = collect(skipmissing(features[!, col]))
                values = filter(x -> x isa Number && !isnan(x), values)
                
                if ! isempty(values)
                    p = histogram(values,
                                title="MFCC $i",
                                xlabel="Value",
                                ylabel="Frequency",
                                legend=false,
                                color=:steelblue,
                                alpha=0.7,
                                bins=50)
                    push!(plots_list, p)
                end
            end
            
            if ! isempty(plots_list)
                p_mfcc = plot(plots_list..., layout=(3,4), size=(1600, 1200))
                display(p_mfcc)
                savefig(p_mfcc, "mfcc_distributions.png")
                println("✓ Saved: mfcc_distributions.png")
            end
        end
    end
    
    # Spectral features
    spectral_patterns = ["centroid", "bandwidth", "rolloff", "contrast"]
    spectral_plots = []
    
    for pattern in spectral_patterns
        matching_cols = filter(x -> occursin(pattern, lowercase(x)) && 
                                    occursin("mean", lowercase(x)), 
                             names(features))
        if !isempty(matching_cols)
            col = matching_cols[1]
            values = collect(skipmissing(features[!, col]))
            values = filter(x -> x isa Number && !isnan(x), values)
            
            if ! isempty(values)
                p = histogram(values,
                            title=uppercasefirst(replace(pattern, "_" => " ")),
                            xlabel="Value",
                            ylabel="Frequency",
                            legend=false,
                            color=:coral,
                            alpha=0.7,
                            bins=50)
                push!(spectral_plots, p)
            end
        end
    end
    
    if !isempty(spectral_plots)
        p_spectral = plot(spectral_plots..., layout=(2,2), size=(1200, 1000))
        display(p_spectral)
        savefig(p_spectral, "spectral_features.png")
        println("✓ Saved: spectral_features.png")
    end
    
    return nothing
end

"""
Correlation analysis between features
"""
function analyze_feature_correlations(features::DataFrame, feature_groups::Dict)
    println("\n" * "="^80)
    println("FEATURE CORRELATION ANALYSIS")
    println("="^80)
    
    # Select numeric columns
    numeric_cols = String[]
    for col in names(features)
        col_type = eltype(features[!, col])
        if col_type <: Union{Missing, Number} || col_type <: Number
            # Check if column has any valid data
            values = collect(skipmissing(features[!, col]))
            values = filter(x -> x isa Number && !isnan(x), values)
            if !isempty(values)
                push!(numeric_cols, col)
            end
        end
    end
    
    if length(numeric_cols) > 1
        # Sample columns
        sample_size = min(50, length(numeric_cols))
        sampled_cols = sample(numeric_cols, sample_size, replace=false)
        
        # Create correlation matrix
        feature_matrix = Matrix{Float64}(undef, nrow(features), length(sampled_cols))
        for (i, col) in enumerate(sampled_cols)
            values = features[!, col]
            clean_values = collect(skipmissing(values))
            clean_values = filter(x -> x isa Number && ! isnan(x), clean_values)
            col_mean = isempty(clean_values) ?  0.0 : mean(clean_values)
            feature_matrix[:, i] = [ismissing(v) || isnan(v) ? col_mean : Float64(v) for v in values]
        end
        
        # Compute correlation
        cor_matrix = cor(feature_matrix)
        
        # Visualization
        p_cor = heatmap(cor_matrix,
                       title="Feature Correlation Matrix (Sample)",
                       xlabel="Feature Index",
                       ylabel="Feature Index",
                       color=:RdBu,
                       clims=(-1, 1),
                       size=(1000, 900))
        display(p_cor)
        savefig(p_cor, "feature_correlations.png")
        println("✓ Saved: feature_correlations.png")
        
        # Find highly correlated pairs
        println("\nHighly Correlated Feature Pairs (|r| > 0.8):")
        high_cor_pairs = []
        for i in 1:length(sampled_cols)
            for j in (i+1):length(sampled_cols)
                if abs(cor_matrix[i, j]) > 0.8 && ! isnan(cor_matrix[i, j])
                    push!(high_cor_pairs, (sampled_cols[i], sampled_cols[j], cor_matrix[i, j]))
                end
            end
        end
        
        if !isempty(high_cor_pairs)
            sorted_pairs = sort(high_cor_pairs, by=x->abs(x[3]), rev=true)
            for (i, (col1, col2, corr)) in enumerate(sorted_pairs[1:min(10, length(sorted_pairs))])
                col1_short = split(col1, "_")[end]
                col2_short = split(col2, "_")[end]
                println("  $i. $col1_short <-> $col2_short: $(round(corr, digits=3))")
            end
        else
            println("  None found")
        end
    end
    
    return nothing
end

#%% ============================================================================
# SECTION 4: ADVANCED MULTIVARIATE ANALYSIS
# ==============================================================================

"""
Principal Component Analysis for dimensionality reduction
"""
function perform_pca_analysis(features::DataFrame, n_components::Int=10)
    println("\n" * "="^80)
    println("PRINCIPAL COMPONENT ANALYSIS")
    println("="^80)
    
    # Prepare data matrix
    numeric_cols = []
    for col in names(features)
        col_type = eltype(features[!, col])
        if col_type <: Union{Missing, Number} || col_type <: Number
            values = collect(skipmissing(features[!, col]))
            values = filter(x -> x isa Number && !isnan(x), values)
            if !isempty(values)
                push!(numeric_cols, col)
            end
        end
    end
    
    if length(numeric_cols) < n_components
        println("Warning: Only $(length(numeric_cols)) features available, adjusting n_components")
        n_components = min(n_components, length(numeric_cols))
    end
    
    if n_components < 2
        println("Error: Not enough features for PCA")
        return nothing, nothing
    end
    
    # Create feature matrix with imputation
    n_samples = min(10000, nrow(features))
    sample_indices = sample(1:nrow(features), n_samples, replace=false)
    
    feature_matrix = Matrix{Float64}(undef, n_samples, length(numeric_cols))
    for (i, col) in enumerate(numeric_cols)
        values = features[sample_indices, col]
        clean_values = collect(skipmissing(values))
        clean_values = filter(x -> x isa Number && !isnan(x), clean_values)
        col_mean = isempty(clean_values) ? 0.0 : mean(clean_values)
        feature_matrix[:, i] = [ismissing(v) || isnan(v) ?  col_mean : Float64(v) for v in values]
    end
    
    # Standardize features
    feature_matrix_std = (feature_matrix .- mean(feature_matrix, dims=1)) ./ (std(feature_matrix, dims=1) .+ 1e-10)
    
    # Perform PCA
    M = fit(PCA, feature_matrix_std', maxoutdim=n_components)
    
    # Explained variance
    explained_var = principalvars(M)./ var(M)
    cumulative_var = cumsum(explained_var)
    
    println("\nExplained Variance by Component:")
    for i in 1:n_components
        println("  PC$i: $(round(explained_var[i] * 100, digits=2))% " *
                "(Cumulative: $(round(cumulative_var[i] * 100, digits=2))%)")
    end
    
    # Visualization
    p1 = plot(1:n_components, explained_var * 100,
             title="Scree Plot",
             xlabel="Principal Component",
             ylabel="Explained Variance (%)",
             marker=:circle,
             markersize=6,
             linewidth=2,
             legend=false,
             color=:steelblue,
             size=(800, 600))
    
    p2 = plot(1:n_components, cumulative_var * 100,
             title="Cumulative Explained Variance",
             xlabel="Number of Components",
             ylabel="Cumulative Variance (%)",
             marker=:circle,
             markersize=6,
             linewidth=2,
             legend=false,
             color=:coral,
             size=(800, 600))
    
    p_pca = plot(p1, p2, layout=(1,2), size=(1600, 600))
    display(p_pca)
    savefig(p_pca, "pca_analysis.png")
    println("✓ Saved: pca_analysis.png")
    
    return M, feature_matrix_std
end

"""
Clustering analysis using K-means
"""
function perform_clustering_analysis(features::DataFrame, n_clusters::Int=8)
    println("\n" * "="^80)
    println("CLUSTERING ANALYSIS")
    println("="^80)
    
    # Prepare data
    numeric_cols = []
    for col in names(features)
        col_type = eltype(features[!, col])
        if col_type <: Union{Missing, Number} || col_type <: Number
            values = collect(skipmissing(features[!, col]))
            values = filter(x -> x isa Number && !isnan(x), values)
            if !isempty(values)
                push!(numeric_cols, col)
            end
        end
    end
    
    if isempty(numeric_cols)
        println("Error: No numeric columns found")
        return nothing
    end
    
    n_samples = min(5000, nrow(features))
    sample_indices = sample(1:nrow(features), n_samples, replace=false)
    
    feature_matrix = Matrix{Float64}(undef, n_samples, length(numeric_cols))
    for (i, col) in enumerate(numeric_cols)
        values = features[sample_indices, col]
        clean_values = collect(skipmissing(values))
        clean_values = filter(x -> x isa Number && ! isnan(x), clean_values)
        col_mean = isempty(clean_values) ?  0.0 : mean(clean_values)
        feature_matrix[:, i] = [ismissing(v) || isnan(v) ? col_mean : Float64(v) for v in values]
    end
    
    # Standardize
    feature_matrix_std = (feature_matrix .- mean(feature_matrix, dims=1))./ (std(feature_matrix, dims=1) .+ 1e-10)
    
    # K-means clustering
    result = kmeans(feature_matrix_std', n_clusters; maxiter=200)
    
    println("\nClustering Results:")
    println("  Number of clusters: $n_clusters")
    println("  Converged: $(result.converged)")
    println("  Iterations: $(result.iterations)")
    
    # Cluster sizes
    cluster_counts = counts(result.assignments)
    println("\nCluster Sizes:")
    for (i, count) in enumerate(cluster_counts)
        pct = round(100 * count / n_samples, digits=2)
        println("  Cluster $i: $count samples ($pct%)")
    end
    
    # Visualization (using first 2 PCA components)
    M = fit(PCA, feature_matrix_std', maxoutdim=2)
    transformed = predict(M, feature_matrix_std')
    
    p_cluster = scatter(transformed[1, :], transformed[2, :],
                       group=result.assignments,
                       title="K-Means Clustering (PCA Projection)",
                       xlabel="PC1",
                       ylabel="PC2",
                       markersize=3,
                       alpha=0.6,
                       size=(1000, 800),
                       legend=:best)
    display(p_cluster)
    savefig(p_cluster, "clustering_analysis.png")
    println("✓ Saved: clustering_analysis.png")
    
    return result
end

#%% ============================================================================
# SECTION 5: GENRE-BASED ANALYSIS
# ==============================================================================

"""
Compare audio features across genres - USING DIRECT INDEXING
"""
function compare_features_by_genre(tracks::DataFrame, features::DataFrame)
    println("\n" * "="^80)
    println("GENRE-BASED FEATURE COMPARISON")
    println("="^80)
    
    # The track IDs are actually the row indices, not a separate column! 
    # Both DataFrames should have the same row indices representing track_id
    
    println("\nNote: Using DataFrame indices as track IDs")
    println("Tracks rows: $(nrow(tracks))")
    println("Features rows: $(nrow(features))")
    
    # Find genre column
    genre_col = find_column(tracks, "genre_top")
    if isnothing(genre_col)
        println("Warning: Cannot find genre column")
        return nothing
    end
    
    # Since both datasets should be aligned by row index, we can directly work with them
    # But first let's check if they have the same number of rows
    if nrow(tracks) != nrow(features)
        println("Warning: Different number of rows in tracks ($(nrow(tracks))) and features ($(nrow(features)))")
        println("Will use only matching rows")
        
        # Use the minimum common rows
        n_rows = min(nrow(tracks), nrow(features))
        tracks_subset = tracks[1:n_rows, :]
        features_subset = features[1:n_rows, :]
    else
        tracks_subset = tracks
        features_subset = features
    end
    
    # Get genre information - filter out missing values first
    genre_data = tracks_subset[!, genre_col]
    valid_indices = findall(x -> !ismissing(x), genre_data)
    
    if isempty(valid_indices)
        println("No valid genre data found")
        return nothing
    end
    
    valid_genres = genre_data[valid_indices]
    genre_counts = countmap(valid_genres)
    top_genres = [x[1] for x in sort(collect(genre_counts), by=x->x[2], rev=true)[1:min(8, length(genre_counts))]]
    
    println("\nAnalyzing top $(length(top_genres)) genres")
    for (i, genre) in enumerate(top_genres)
        println("  $i. $genre: $(genre_counts[genre]) tracks")
    end
    
    # Find representative features
    feature_patterns = ["mfcc_mean", "centroid_mean", "bandwidth_mean", "zcr"]
    available_features = []
    
    for pattern in feature_patterns
        col = find_column(features_subset, pattern)
        if ! isnothing(col)
            push!(available_features, col)
        end
    end
    
    if isempty(available_features)
        println("\nNo matching features found.  Trying alternative patterns...")
        # Try to find any mean features
        mean_features = filter(x -> occursin("mean", lowercase(x)), names(features_subset))
        if !isempty(mean_features)
            available_features = mean_features[1:min(4, length(mean_features))]
            println("Using features: $(join(available_features, ", "))")
        else
            println("Could not find suitable features for comparison")
            return nothing
        end
    end
    
    available_features = available_features[1:min(4, length(available_features))]
    
    println("\nComparing $(length(available_features)) features across genres")
    
    # Create plots
    plots_list = []
    for feat in available_features
        try
            genre_values = []
            genre_labels = []
            
            for genre in top_genres
                # Find indices where genre matches (handling missing values)
                genre_indices = findall(i -> !ismissing(genre_data[i]) && genre_data[i] == genre, 1:length(genre_data))
                
                if !isempty(genre_indices)
                    values = features_subset[genre_indices, feat]
                    values = collect(skipmissing(values))
                    values = filter(x -> x isa Number && !isnan(x), values)
                    
                    if !isempty(values)
                        append!(genre_values, values)
                        append!(genre_labels, fill(string(genre), length(values)))
                    end
                end
            end
            
            if !isempty(genre_values) && length(unique(genre_labels)) > 1
                feat_name = split(feat, "_")[1]
                p = boxplot(genre_labels, genre_values,
                           title="$feat_name by Genre",
                           xlabel="Genre",
                           ylabel="Value",
                           legend=false,
                           xrotation=45,
                           size=(800, 600),
                           bottom_margin=10Plots.mm)
                push!(plots_list, p)
                println("  ✓ Created plot for $feat")
            else
                println("  ⚠ Skipping $feat - insufficient data")
            end
        catch e
            println("  ✗ Error plotting $feat: $e")
            # Print more detailed error info
            if isdefined(Main, :InteractiveUtils)
                showerror(stdout, e, catch_backtrace())
                println()
            end
        end
    end
    
    if !isempty(plots_list)
        n_plots = length(plots_list)
        layout_rows = Int(ceil(n_plots / 2))
        layout_cols = min(2, n_plots)
        
        p_genre_comp = plot(plots_list..., 
                           layout=(layout_rows, layout_cols), 
                           size=(layout_cols * 800, layout_rows * 600))
        display(p_genre_comp)
        savefig(p_genre_comp, "genre_feature_comparison.png")
        println("\n✓ Saved: genre_feature_comparison. png")
        println("Successfully created $(n_plots) comparison plots")
    else
        println("\nCould not create any comparison plots")
    end
    
    return nothing
end
#%% ============================================================================
# SECTION 6: SUMMARY AND CONCLUSIONS
# ==============================================================================

"""
Generate comprehensive summary report
"""
function generate_summary_report(tracks::DataFrame, features::DataFrame)
    println("\n" * "="^80)
    println("COMPREHENSIVE DATA SUMMARY")
    println("="^80)
    
    # Count unique genres
    genre_col = find_column(tracks, "genre_top")
    n_genres = 0
    if !isnothing(genre_col)
        valid_genres = collect(skipmissing(tracks[!, genre_col]))
        n_genres = length(unique(valid_genres))
    end
    
    summary = """
    # FMA Dataset Exploratory Data Analysis Summary
    
    ## Dataset Overview
    - **Tracks**: $(nrow(tracks)) tracks
    - **Features**: $(ncol(features)) audio features extracted
    - **Genres**: $n_genres unique top-level genres
    
    ## Key Findings
    
    ### 1. Data Quality
    - Comprehensive multi-level metadata structure
    - Rich audio feature extraction (MFCC, spectral, temporal)
    - Some missing values in metadata fields
    
    ### 2.  Audio Features
    - High-dimensional feature space with $(ncol(features)) features
    - Multiple feature types: MFCC, chroma, spectral, temporal
    - Features show expected correlations within groups
    
    ### 3.  Data Characteristics
    - Large-scale dataset suitable for machine learning
    - Diverse genre representation
    - Rich metadata including user engagement metrics
    
    ## Recommendations
    
    1. **For Classification Tasks**:
       - Use dimensionality reduction (PCA) to handle high-dimensional features
       - Consider class imbalance in genre distribution
       - Implement cross-validation strategies
    
    2. **For Clustering/Recommendation**:
       - Leverage audio features for similarity metrics
       - Combine content-based and collaborative filtering
       - Explore hierarchical genre structure
    
    3. **For Further Analysis**:
       - Investigate artist and album effects
       - Perform temporal trend analysis
       - Explore feature engineering opportunities
    
    ## Generated Visualizations
    - Genre distribution
    - Duration statistics
    - Engagement metrics
    - Feature distributions and correlations
    - PCA and clustering analysis
    - Genre-based feature comparisons
    """
    
    println(summary)
    
    # Save report
    open("eda_summary_report.md", "w") do file
        write(file, summary)
    end
    
    println("\n✓ Summary report saved to 'eda_summary_report.md'")
    
    return summary
end