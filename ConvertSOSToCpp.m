% MATLAB Script: Convert SOS matrix to C++ code format for MTIFilter
% This script converts a Second-Order Sections (SOS) matrix into a C++ 
% static float array suitable for the MTIFilter class.

% Check if SOS exists in workspace
if ~exist('SOS', 'var')
    error('Variable "SOS" not found in workspace. Please generate or load it first.');
end

% Determine matrix size
[num_sections, num_cols] = size(SOS);

% Check if it is a standard SOS format
if num_cols ~= 6
    error('SOS matrix should have 6 columns (b0, b1, b2, a0, a1, a2)');
end

% Incorporate Gain vector G if it exists
if exist('G', 'var')
    fprintf('Variable "G" found. Incorporating gains into SOS b-coefficients...\n');
    % Typically G has num_sections + 1 elements
    % The last element is the overall gain, often applied to the first or last section
    % Or G has num_sections elements, one for each section
    lenG = length(G);
    if lenG == num_sections
        for i = 1:num_sections
            SOS(i, 1:3) = SOS(i, 1:3) * G(i);
        end
    elseif lenG == num_sections + 1
        % Apply section gains
        for i = 1:num_sections
            SOS(i, 1:3) = SOS(i, 1:3) * G(i);
        end
        % Apply overall gain (G(end)) to the first section
        SOS(1, 1:3) = SOS(1, 1:3) * G(end);
    else
        fprintf('Warning: Length of G (%d) does not match num_sections (%d). G ignored.\n', lenG, num_sections);
    end
end

% Create C++ code string
cpp_code = sprintf('static const float SOS[%d][6] = {\n', num_sections);

% Add all rows
for i = 1:num_sections
    cpp_row = sprintf('    {%.6ff, %.6ff, %.6ff, %.6ff, %.6ff, %.6ff}', ...
                        SOS(i,1), SOS(i,2), SOS(i,3), SOS(i,4), SOS(i,5), SOS(i,6));
    
    cpp_code = [cpp_code, cpp_row]; %#ok<AGROW>
    
    % If not the last row, add a comma
    if i < num_sections
        cpp_code = [cpp_code, sprintf(',\n')]; %#ok<AGROW>
    else
        cpp_code = [cpp_code, sprintf('\n')]; %#ok<AGROW>
    end
end

% Complete array definition
cpp_code = [cpp_code, sprintf('};\n')];

% Display result
fprintf('\nCopy the following code to your C++ MTIFilter::apply function:\n\n');
fprintf('%s', cpp_code);

% Also copy to clipboard for convenience
try
    clipboard('copy', cpp_code);
    fprintf('\n(The code has also been copied to your clipboard)\n');
catch
    % Clipboard might fail in some environments
end
