% this piece of code was taken from the internet, author unknown
function [out] = cgr(chars, order, k)
    out = zeros(2^k);
    x = 2^(k-1);
	y = 2^(k-1);

    for i = 1:length(chars)
        char = chars(i);

        x = fix(x/2);
        if char == order(3) || char == order(4)
            x = x + 2^(k-1);
        end

        y = fix(y/2);
        if char == order(1) || char == order(4)
            y = y + 2^(k-1);
        end

        if i >= k
            out(y+1, x+1) = out(y+1, x+1) + 1;
        end
    end
end
