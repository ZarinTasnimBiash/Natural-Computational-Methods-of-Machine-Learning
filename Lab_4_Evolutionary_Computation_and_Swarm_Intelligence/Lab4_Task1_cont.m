ack = GAparams;
ack.visual.type = 'mesh';
%ga_visual_ackley([],[],[],[],[],[],ack.visual,[],[]);

ack.visual.bounds = [-2, 2];
ack.visual.interval = 0.05;
%ga_visual_ackley([],[],[],[],[],[],ack.visual,[],[]);

ack.stop.direction = 'min';
ack.visual.func = 'ackley';
ack.visual.active = true;
%[best, fit, stat] = GAsolver(20, [-32, 32], ...
%'ackley', 200, 250, ack);
% 
% crossover_funcs = {'1point', 'blend', 'arithmetic'};
% mutate_decays = {'none', 'linear', 'exponential'};
% comparative_opts = [false, true];
% 
% combination_count = 0;
% best_overall_fitness = Inf; % Since you're minimizing
% best_combo = struct();      % To store the best config
% 
% for i = 1:length(crossover_funcs)
%     for j = 1:length(mutate_decays)
%         for k = 1:length(comparative_opts)
%             if combination_count == 18
%                 break;
%             end
%             combination_count = combination_count + 1;
%             fprintf('Run %d: crossover=%s, decay=%s, comparative=%d\n', ...
%                 combination_count, crossover_funcs{i}, mutate_decays{j}, comparative_opts(k));
% 
%             ack = GAparams;
%             ack.crossover.func = crossover_funcs{i};
%             ack.mutate.decay = mutate_decays{j};
%             ack.replace.comparative = comparative_opts(k);
%             ack.stop.direction = 'min';
%             ack.visual.active = true;
% 
%             [best, fit, stat] = GAsolver(20, [-32 32], 'ackley', 200, 250, ack);
% 
%             current_best = fit(1); % best fitness from this run
% 
%             % Store results
%             results{combination_count}.crossover = crossover_funcs{i};
%             results{combination_count}.decay = mutate_decays{j};
%             results{combination_count}.comparative = comparative_opts(k);
%             results{combination_count}.best_fit = current_best;
% 
%             % Check if this is the best so far
%             if current_best < best_overall_fitness
%                 best_overall_fitness = current_best;
%                 best_combo.crossover = crossover_funcs{i};
%                 best_combo.decay = mutate_decays{j};
%                 best_combo.comparative = comparative_opts(k);
%                 best_combo.best_fit = current_best;
%             end
%         end
%         if combination_count == 18
%             break;
%         end
%     end
%     if combination_count == 18
%         break;
%     end
% end
% 
% % Display the best combination found
% fprintf('\nBest configuration:\n');
% fprintf('Crossover: %s\n', best_combo.crossover);
% fprintf('Mutation decay: %s\n', best_combo.decay);
% fprintf('Comparative replacement: %d\n', best_combo.comparative);
% fprintf('Best fitness: %.4f\n', best_combo.best_fit);
% 
% % Plot the best run's diversity curve
% ga_plot_diversity(stat); % This uses the last run's stat — best if it's the best run too
% 
% % Add title with best combo details
% plot_title = sprintf('Best: crossover = %s, decay = %s, comparative = %d, fitness = %.4f', ...
%     best_combo.crossover, best_combo.decay, best_combo.comparative, best_combo.best_fit);
% title(plot_title, 'Interpreter', 'none');

% Define the number of trials per setting
num_trials = 5; % Run each configuration multiple times

% Define parameter combinations
crossover_funcs = {'arithmetic', 'blend', 'linear'};
mutate_decays = {'none', 'linear', 'exponential'};
mutate_proportional_options = [true, false];

% Store results
results = [];
index = 1;

for c = 1:length(crossover_funcs)
    for m = 1:length(mutate_decays)
        for p = 1:length(mutate_proportional_options)
            
            ack = GAparams;

            ack.stop.direction = 'min';
            ack.visual.func = 'ackley';
            ack.visual.active = false; % Disable visualization for faster execution
            
            ack.crossover.func = crossover_funcs{c};
            ack.mutate.decay = mutate_decays{m};
            ack.replace.comparative = mutate_proportional_options(p);
            
            fitness_values = zeros(1, num_trials);
            
            for j = 1:num_trials
                [best, fit, stat] = GAsolver(20, [-32 32], 'ackley', 200, 250, ack);
                fitness_values(j) = fit(1);
            end
            
            avg_fitness = mean(fitness_values);
            
            % Store the result in a struct
            results(index).crossover_func = crossover_funcs{c};
            results(index).mutate_decay = mutate_decays{m};
            results(index).replace_comparative= mutate_proportional_options(p);
            results(index).average_fitness = avg_fitness;
            index = index + 1;
        end
    end
end

% Convert results to a table for better visualization
results_table = struct2table(results);

% Display results
disp(results_table);