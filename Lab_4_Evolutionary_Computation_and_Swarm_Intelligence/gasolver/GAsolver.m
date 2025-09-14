function [solution, varargout] = GAsolver(len, bounds, fitnessfunc, varargin)
%GAsolver a matrix-operation based genetic algorithms solver
%   len             the length of each chromosome
%   bounds          len-by-2 matrix of upper, lower bounds for each gene
%   fitnessfunc     string name of the object function.
%                   The object function must take a popSize by len matrix
%                   of genomes, and return a len vector of objective values

%  code is modified from: SpeedyGA, Version 1.3
%  Copyright (C) 2007, 2008, 2009  Keki Burjorjee
%  Created and tested under Matlab 7 (R14). 

p = inputParser;
p.addRequired('len', @(x)isnumeric(x) && isscalar(x) && x>0);
p.addRequired('bounds', @isnumeric);
p.addRequired('fitnessfunc', @ischar);

% The size of the population (must be an even number)
p.addOptional('popSize', 500, @(x)isnumeric(x) && isscalar(x) && x>0);

% The maximum number of generations allowed in a run
p.addOptional('maxGens', 1000, @(x)isnumeric(x) && isscalar(x) && x>0);
    
% Probability distribution used to generate the initial population
init_defaults = struct('func','default','variance',1,'peak','mean');
init_func_default = 'uniform';
p.addParamValue('init',init_defaults,@isstruct);

% The probability of crossing over.
crossover_defaults = struct('func','default','prob',0.9);
crossover_func_default = '1point';
p.addParamValue('crossover',crossover_defaults,@isstruct);

mutate_defaults = struct('func','default','prob',0.003,'step',0.1,'decay','none','proportional',false);
mutate_func_default = 'uniform';
p.addParamValue('mutate',mutate_defaults,@isstruct);

replace_defaults = struct('func','default','elitist',true,'gap',0,'comparative',true);
replace_func_default = 'worst';
p.addParamValue('replace',replace_defaults,@isstruct);
% Scaling function applied to the raw fitness value:
% Sigma Scaling is described on [2], p. 168
validScalingTypes = {'none', 'sigma'};
p.addParamValue('scalingFunc', 'none', @(x)any(strcmp(x,validScalingTypes)));
% Higher values => less fitness pressure
p.addParamValue('scalingCoeff', 1, @(x)isnumeric(x) && isscalar(x) && x>=0 && x<=1);

% attempt to visualize evolution
visual_defaults = struct('func','default','type','contour','step',10,'active',false);
visual_func_default = 'default';
p.addParamValue('visual',visual_defaults,@isstruct);

% display numeric values each generation
p.addParamValue('verbose', true, @(x)islogical(x) && isscalar(x));

validgeneTypes = {'float', 'binary'};
p.addParamValue('geneType', 'float', @(x)any(strcmp(x,validgeneTypes)));

% Selection operator for crossover:
select_defaults = struct('func','default','sampling','roulette','pressure',2.0);
select_func_default = 'proportional';
p.addParamValue('select', select_defaults, @isstruct);

% an object that is passed to the objective function on each iteration:
p.addParamValue('objParams', struct());

stop_defaults = struct('func','default','direction','max','window',100,'error',0.005);
stop_func_default = 'generations';
p.addParamValue('stop', stop_defaults, @isstruct);

p.addParamValue('diversity', 'moi', @ischar);

p.StructExpand = true;
p.KeepUnmatched = true;
p.parse(len, bounds, fitnessfunc, varargin{:});

% Set default values for missing fields
    function [updated, ga_func] = ga_struct_defaults(name)
        updated = p.Results.(name);
        default = eval([name '_defaults']);
        f = fieldnames(default);
        for i=1:length(f)
            if ~isfield(updated,f{i})
                updated.(f{i}) = default.(f{i});
            end
        end
        if strcmp('default',updated.func)
            if exist(['ga_' name '_' p.Results.fitnessfunc])
                ga_func = str2func(['ga_' name '_' p.Results.fitnessfunc]);
                updated.func = p.Results.fitnessfunc;
            else
                ga_func = str2func(['ga_' name '_' eval([name '_func_default'])]);
                updated.func = eval([name '_func_default']);
            end
        else
            ga_func = str2func(['ga_' name '_' updated.func]);
        end
    end
[init_params, init_func] = ga_struct_defaults('init');
[select_params, select_func] = ga_struct_defaults('select');
[crossover_params, crossover_func] = ga_struct_defaults('crossover');
[replace_params, replace_func] = ga_struct_defaults('replace');
[mutate_params, mutate_func] = ga_struct_defaults('mutate');
[visual_params, visual_func] = ga_struct_defaults('visual');
[stop_params, stop_func] = ga_struct_defaults('stop');
obj_func = str2func(['ga_fit_' fitnessfunc]);
if ~strcmp('none',p.Results.diversity)
    div_func = str2func(['ga_div_' p.Results.diversity]);
    calcdiv = true;
else
    calcdiv = false;
end
fprintf('\n')
disp 'List of all arguments:';
ga_show_parameters(p.Results);
if (~isempty(p.Unmatched))
    disp 'Unrecognized parameters:';
    display(p.Unmatched);
end
tic;


% if only one upper and lower bound was supplied, apply it to all locii
if (size(bounds,1) == 1)
    bounds = repmat(bounds,len,1);
end


% preallocate vectors for recording the average and maximum fitness in each
% generation
fitnessHist = zeros(4,p.Results.maxGens+1);
%minFitnessHist=zeros(1,p.Results.maxGens+1);
%avgFitnessHist=zeros(1,p.Results.maxGens+1);
%maxFitnessHist=zeros(1,p.Results.maxGens+1);

diversity = zeros(1,p.Results.maxGens+1);

eliteIndiv=[];
eliteFitness=[];
eliteDob=[];
eliteMob=[];

% the population is a popSize by len matrix of randomly generated values

pop = init_func(p.Results.popSize, len, bounds, init_params, p.Results.objParams);

if visual_params.active
    h = figure();
end

staticFitnessGens = 0;

% Determine number of offspring for each generation
% The generation gap gives the proportion of the current population that
% should survive to the next
if replace_params.gap == 0  %
    numOffspring = p.Results.popSize;
    %replace_func = @ga_replace_all;
else
    numOffspring = ceil((1 - replace_params.gap) * p.Results.popSize);
    if mod(numOffspring,2) == 1
        if (numOffspring < p.Results.popSize)
            numOffspring = numOffspring + 1;
        else
            numOffspring = numOffspring - 1;
        end
    end
end
replace_params.direction = stop_params.direction;

% Compute fitness of the initial population
fitnessVals=obj_func(pop, bounds, p.Results.objParams);

dob = zeros(1, p.Results.popSize);
mob = zeros(1, p.Results.popSize);
methods = {'initialization', 'crossover', 'mutation'};

for gen=0:p.Results.maxGens
    % Record population statistics
    fitnessHist(2,gen+1)=mean(fitnessVals);
    
    switch stop_params.direction
        case 'max'
            [sortedFitness, rank] = sort(fitnessVals, 'ascend');
        case 'min'
            [sortedFitness, rank] = sort(fitnessVals, 'descend');
    end
    fitnessHist(3,gen+1)= sortedFitness(1);    
    fitnessHist(1,gen+1)= sortedFitness(end);
    fitnessHist(4,gen+1)= std(fitnessVals);
    
    if calcdiv
        diversity(1,gen+1) = div_func(pop, fitnessVals);
    end
    
    % Update history of elite individuals
    switch stop_params.direction
        case 'max'
            if isempty(eliteFitness) || eliteFitness(1) < fitnessHist(1,gen+1)
                eliteFitness=[sortedFitness(end) eliteFitness];
                eliteIndiv=[pop(rank(p.Results.popSize),:) ; eliteIndiv];
                eliteDob=[dob(rank(p.Results.popSize))  eliteDob];
                eliteMob=[mob(rank(p.Results.popSize))  eliteMob];
                staticFitnessGens = 0;
            else
                staticFitnessGens = staticFitnessGens + 1;
            
            end
        case 'min'
            if isempty(eliteFitness) || eliteFitness(1) > fitnessHist(1,gen+1)
                eliteFitness=[sortedFitness(end) eliteFitness];
                eliteIndiv=[pop(rank(p.Results.popSize),:) ; eliteIndiv];
                eliteDob=[dob(rank(p.Results.popSize)) eliteDob];
                eliteMob=[mob(rank(p.Results.popSize))  eliteMob];
                staticFitnessGens = 0;
            else
                staticFitnessGens = staticFitnessGens + 1;
            end
    end
    
    
    % Display the generation number, the average Fitness of the population,
    % and the maximum fitness of any individual in the population
    if p.Results.verbose
        display(['gen=' num2str(gen,'%.3d') '   Fitness' ...
            '  worst=' num2str(fitnessHist(3,gen+1),'%3.4f') ...
            '  mean=' num2str(fitnessHist(2,gen+1),'%3.4f') ... 
            '  best=' num2str(fitnessHist(1,gen+1),'%3.4f')]);
    end
    
    % Conditionally perform fitness scaling 
    switch p.Results.scalingFunc
        case 'sigma'
            sigma=std(fitnessVals);
            if sigma~=0;
                fitnessVals=1+(fitnessVals-mean(fitnessVals))/...
                (p.Results.scalingCoeff*sigma);
                fitnessVals(fitnessVals<=0)=0;
            else
                fitnessVals=ones(p.Results.popSize,1);
            end
    end        
    
    %-----------
    % CROSSOVER
    [offspring, offspringFit, offspringDob, offspringMob, parentIndices] =...
        crossover_func(pop, rank, fitnessVals, dob, mob, eliteFitness, numOffspring, gen,...
        select_func, select_params, stop_params.direction, obj_func, crossover_params);
    
    
    %----------
    % MUTATION
    
    switch mutate_params.decay
        case 'exponential'
            mutProb  = mutate_params.prob + .11375 / 2^gen;
        case 'linear'
            mutProb = mutate_params.prob * (p.Results.maxGens - gen) / p.Results.maxGens;
        otherwise
            mutProb = mutate_params.prob;
    end
    
    if mutate_params.proportional
        % need to know unmutated fitness before we can decide proportions
        offspringFit(offspringFit == -inf)=obj_func(offspring(offspringFit == -inf,:), bounds, p.Results.objParams);
        propFitness = zeros(1,numOffspring);
        switch stop_params.direction
            case 'max'
                propFitness = 1 ./ (1 + eliteFitness(1) - offspringFit);
            case 'min'
                propFitness = 1 ./ (1 + offspringFit - eliteFitness(1));
        end
        normFitnessVals= 1 - propFitness;
        mutProb = repmat(normFitnessVals' * mutProb,1,len);
    end
    
    [offspring, offspringFit, offspringDob, offspringMob] = mutate_func(offspring, offspringFit, offspringDob, offspringMob, mutProb, gen, p.Results.geneType, mutate_params);

    %------------
    % REPLACEMENT

    % Make sure all genes are in bounds:
    for g=1:len
        pop(pop(:,g) > bounds(g,2),g) = bounds(g,2);
        pop(pop(:,g) < bounds(g,1),g) = bounds(g,1);
        offspring(offspring(:,g) > bounds(g,2),g) = bounds(g,2);
        offspring(offspring(:,g) < bounds(g,1),g) = bounds(g,1);
    end
    
    
    % Fitness of the newly generated population (i.e., any fitness which
    % seems to be -inf).
    % Fitness of the survivors is not recalculated.
    if strcmp(p.Results.scalingFunc,'none')
        offspringFit(offspringFit == -inf)=obj_func(offspring(offspringFit == -inf,:), bounds, p.Results.objParams);
    else
        offspringFit = obj_func(offspring, bounds, p.Results.objParams);
    end
    
    [ pop, fitnessVals, dob, mob ] = replace_func( pop, fitnessVals, dob, mob, rank, offspring, offspringFit, offspringDob, offspringMob, parentIndices, gen, replace_params);

    %--------------
    % FINISHING UP

    if visual_params.active
        if (mod(gen, visual_params.step) == 0)
            visual_func(bounds, gen, pop, fitnessVals, eliteIndiv, eliteFitness, visual_params, p.Results.objParams, h);
        end
    end
    
    
    elapsed = toc;
    [term_string, finished] = stop_func(gen, fitnessVals, staticFitnessGens, elapsed, stop_params);
    
    if finished
        break;
    end 
end
    
if p.Results.verbose
    display(['Time: ' num2str(elapsed)]);
    display(['Termination: ' term_string]);
    display(['Best individual had a fitness of ' num2str(eliteFitness(1), '%.4f')]);
    switch p.Results.geneType
        case 'float'
            display(['    at [ ' sprintf('%.4f ', eliteIndiv(1,:)) ']']);
        otherwise
            display(['    at [ ' sprintf('%.0f ', eliteIndiv(1,:)) ']']);    
    end
    display(['    and was born in generation ' num2str(eliteDob(1)) ' via ' methods{eliteMob(1) + 1} ]);
end
stats = struct('diversity', diversity, 'time', elapsed, 'generations', gen, 'termination', term_string, 'eliteInd', eliteIndiv, 'eliteFit', eliteFitness, 'eliteDob', eliteDob, 'eliteMob', eliteMob, 'fitness', fitnessHist);
solution = eliteIndiv(1,:);
varargout = { eliteFitness(1), stats };
varargout = { varargout{1:nargout - 1} };
end