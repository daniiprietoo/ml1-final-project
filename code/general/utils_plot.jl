using Plots
using DataFrames
using StatsPlots

function draw_results(x, y; colors, target_names=nothing)
    num_classes = length(unique(colors))

    if !isnothing(target_names)
        @assert num_classes == length(target_names)
        label = target_names
    else
        label = [string(i) for i in 1:num_classes]
    end

    fig = plot()
    if (num_classes == 2)
        possitive_class = y[:,1].==1
        scatter!(fig, x[possitive_class,1], x[possitive_class,2], markercolor=colors[1], label=label[1])
        scatter!(fig, x[.!possitive_class,1], x[.!possitive_class,2], markercolor=colors[2], label=label[2])
    else
        for i in 1:num_classes
            index_class = y[:,i].==1
            scatter!(fig, x[index_class, 1], x[index_class, 2], markercolor=colors[i], label=label[i])
        end
    end

    return fig;
end

function plot_model_comparison(results_df::DataFrame; title_str::String="Best Model Performance: Accuracy")
    best_results = combine(groupby(results_df, :Model), :Accuracy => maximum => :Accuracy)
    
    best_results.Model = string.(best_results.Model)
    
    sort!(best_results, :Accuracy, rev=true)

    p = @df best_results bar(
        :Model,
        :Accuracy,
        title = "Best Accuracy by Model Type",
        xlabel = "Model",
        ylabel = "Accuracy",
        legend = false,
        color = :cornflowerblue,
        xrotation = 45,
        ylims = (0, 1.05), # Slightly above 1.0 to see the top of the bars
        hover = :Accuracy # Show value on hover
    )
    
    # 5. Save and Display
    savefig(p, "model_comparison.png")
    display(p)
end

function plot_grouped_comparison(results_df::DataFrame; title_str::String="Best Model Performance: Accuracy vs F1-Score")
    best_results = combine(groupby(results_df, :Model)) do df
        df[argmax(df.Accuracy), :]
    end
    
    best_results.Model = string.(best_results.Model)
    
    sort!(best_results, :Accuracy, rev=true)

    long_df = stack(best_results, [:Accuracy, :F1], variable_name=:Metric, value_name=:Value)

    p = @df long_df groupedbar(
        :Model,
        :Value,
        group = :Metric,
        title = title_str,           
        xlabel = "Model Architecture",
        ylabel = "Score (0-1)",
        bar_position = :dodge,   
        bar_width = 0.7,
        legend = :outertopright,
        color = [:navy :orange],
        ylims = (0, 1.1),
        xrotation = 45,
        framestyle = :box,
        grid = :y
    )
    
    hline!(p, [0.333], linestyle=:dash, color=:gray, label="Random Baseline", linewidth=2)

    safe_filename = replace(title_str, " " => "_") * ".png"
    savefig(p, "plots/$safe_filename")
    display(p)
end

function plot_tradeoff_scatter(results_df::DataFrame; title_str::String="Trade-off: Accuracy vs Sensitivity")
    df_plot = copy(results_df)
    df_plot.Model = string.(df_plot.Model)

    p = @df df_plot scatter(
        :Sensitivity,
        :Accuracy,
        group = :Model,          # Color points by Model type
        title = title_str,       
        xlabel = "Sensitivity (Recall)",
        ylabel = "Accuracy",
        markersize = 6,
        markeralpha = 0.7,
        legend = :outertopright,
        xlims = (0, 1.05),
        ylims = (0, 1.05),
        framestyle = :box
    )
    
    plot!(p, [1], [1], marker=:star, markersize=10, color=:gold, label="Perfect Model")

    safe_filename = replace(title_str, " " => "_") * ".png"
    savefig(p, "plots/$safe_filename")
    display(p)
end