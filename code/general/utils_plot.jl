using Plots;

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