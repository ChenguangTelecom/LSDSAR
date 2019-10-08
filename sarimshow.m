function [] = sarimshow(u,varargin)
    %% Input Description 
    %
    %   + u : real image (typically, the modulus of a SAR image)
    %
    % list of optional inputs: 'title','nsig'
    %
    %   + 'title' (followed by an argument of type 'string'): set specified
    %             frame title. 
    %
    %   + 'nsig' (followed by an argument of type 'scalar'): set gray-level
    %             threshold = mean(u(:)) + nsig*std(u(:)). 
    %             DEFAULT: nsig = 3.
    %
    %   + 'thresh' (followed by an argument of type 'scalar'): use a
    %             specified threshold (instead of average+nsig*stdev).
    %
    %   + 'showsaturation' (followed by a boolean argument): if set to
    %             true, display in red the saturated values.
    %  
    %% parse inputs 
    p = inputParser;
    p.addRequired('mainVector',@(u) (length(u)>1)&&(isreal(u)));
    p.addOptional('nsig',-1,@isscalar);
    p.addOptional('thresh',-1,@isscalar);
    p.addOptional('title','',@isstr);
    p.addOptional('showsaturation',false,@islogical);
    p.parse(u,varargin{:})
    
    %% set threshold value
    v = double(u); 
    if (p.Results.nsig == -1) && (p.Results.thresh == -1)
        val_max = mean(v(:)) + 3*std(v(:)); 
    elseif (p.Results.nsig ~= -1) && (p.Results.thresh ~= -1)
        error('select only one option between ''nsig'' and ''thresh''');
    elseif (p.Results.nsig ~= -1)
        val_max = mean(v(:)) + p.Results.nsig*std(v(:));
    elseif (p.Results.thresh ~= -1)
        val_max = p.Results.thresh; 
    end
    fprintf('max. displayed value = %.16g\n',val_max);
    
    %% display image
    v(v > val_max) = val_max; 
    m = min(v(:)); 
    imshow((u-m)/(val_max-m)); 
    if ~strcmp(p.Results.title,'')
        set(gcf,'Name',p.Results.title); 
    end
    if p.Results.showsaturation
        fg = gcf(); 
        fg.Colormap(end,:) = [1,0,0]; 
    end
    
end

