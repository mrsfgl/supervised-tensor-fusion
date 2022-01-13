function im = generate_image(sz, ratio, varargin)
%% im = generate_image(sz)
% Generates a dummy image for simulation.

if nargin >= 3
    modes = varargin{1};
    M = length(modes);
else
    M = 1;
    modes = {1:length(sz)};
end
im = cell(M,1);
if nargin < 4
    for i=1:M
        sizes = sz(modes{i});
        if length(sizes) == 3
            %%%%%%%%%%%%%%% Generate Image 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            m = zeros(sizes);
            m(round((sizes(1)+(-2:ratio:2))/ratio),round((sizes(2)+(-10:ratio:10))/ratio),...
                round((sizes(3)+(-2:ratio:2))/ratio)) = 1;
            m(round((sizes(1)+(-10:ratio:10))/ratio),round((sizes(2)+(-2:ratio:2))/ratio),...
                round((sizes(3)+(-2:ratio:2))/ratio)) = 1;
            m(round((sizes(1)+(-2:ratio:2))/ratio),round((sizes(2)+(-2:ratio:2))/ratio),...
                round((sizes(3)+(-10:ratio:10))/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round((6:ratio:10)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round(end+(-10:ratio:-6)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round((6:ratio:10)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round(end+(-10:ratio:-6)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round((6:ratio:10)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round(end+(-10:ratio:-6)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round((6:ratio:10)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round(end+(-10:ratio:-6)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
        else
            m = zeros(sizes);
            m(round((sizes(1)+(-2:ratio:2))/ratio),round((sizes(2)+(-10:ratio:10))/ratio)) = 1;
            m(round((sizes(2)+(-10:ratio:10))/ratio),round((sizes(1)+(-2:ratio:2))/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round((6:ratio:10)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round((6:ratio:10)/ratio)) = 1;
            m(round(end+(-10:ratio:-6)/ratio),round(end+(-10:ratio:-6)/ratio)) = 1;
        end
        im{i} = m;
    end
else
    rank = varargin{2};
    for j = 1:M
        im{j} = zeros(sz(modes{j}));
    end
    for r = 1:rank
        U = cell(length(sz), 1);
        for i = 1:length(sz)
            U{i} = randn(sz(i),1);
            U{i} = U{i}/norm(U{i});
        end
        for j = 1:M
            im{j} = im{j} + reshape(getouter(U(modes{j})), sz(modes{j}));
        end
    end
end

end