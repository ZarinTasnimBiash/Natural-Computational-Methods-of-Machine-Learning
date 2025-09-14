function f = nntrainfcn( pop )
%NNTRAINFCN Summary of this function goes here
%   Detailed explanation goes here

    global psonet
    global P
    global T
    if strcmp(pop,'init')
        f.PopInitRange = [-1; 1] ;
    else
        f = zeros(size(psonet,1));
        for i=1:size(pop,1)
            psonet = setx(psonet,pop(i,:)');
            Y = psonet(P);
            f(i) = mse(T-Y);
        end
    end

end

