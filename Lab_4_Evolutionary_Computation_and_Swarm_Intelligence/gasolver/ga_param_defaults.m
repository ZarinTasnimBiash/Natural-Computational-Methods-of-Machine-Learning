function [ newparam ] = ga_param_defaults( param, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    newparam = param;
    for i=1:length(varargin)/2
        if ~isfield(newparam, varargin{2*i-1})
            newparam = setfield(newparam,varargin{2*i-1},varargin{2*i});
        end
    end
end
    
