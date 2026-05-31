% MATLAB Script: Design Pilot Positions
% Calculates pilot positions based on FFT size and desired number of pilots.
% Output is formatted for C++ vector initialization.

N_pilot=16; % Number of pilots
i=1:N_pilot;
% Calculate pilot positions before IFFT shift (uniform spacing)
pilots_before_ifftshift = 1024/(N_pilot+1)*i-1;
pilots_before_ifftshift = round(pilots_before_ifftshift);

% Apply IFFT shift to get actual subcarrier indices
pilots_after_ifftshift = mod(pilots_before_ifftshift + 512, 1024);

% Print the pilot positions in C++ format
fprintf('_pilot_positions{');
for i = 1:length(pilots_after_ifftshift)
    if i == length(pilots_after_ifftshift)
        fprintf('%d', pilots_after_ifftshift(i));
    else
        fprintf('%d, ', pilots_after_ifftshift(i));
    end
end
fprintf('},\n\n');