
%Task 1 : reading the image
x = imread("img1.JPG");
imshow(x);

%Task 2 : converting image to gray scale
red = x(:, :, 1);
green = x(:, :, 2);
blue = x(:, :, 3);

gray = .299*red + .587*blue + .114*blue ;

imshow(gray);

%Task 3 : adding s&p noise to image

% Define the percentage of pixels to be contaminated with salt and pepper noise
salt_pepper_ratio = 0.3;

% Calculate the number of pixels to be contaminated
num_salt_pepper_pixels = round(salt_pepper_ratio * numel(gray));

% Generate random coordinates for salt and pepper noise
salt_pepper_indices = randperm(numel(gray), num_salt_pepper_pixels);

% Add salt and pepper noise
for k = 1:num_salt_pepper_pixels
    % Randomly decide whether to add salt or pepper noise
    if rand < 0.5
        % Salt noise: set pixel value to maximum (255)
        gray(salt_pepper_indices(k)) = 255;
    else
        % Pepper noise: set pixel value to minimum (0)
        gray(salt_pepper_indices(k)) = 0;
    end
end

imshow(gray);


%Task 4 : Removing the noise using Adaptive median filter : 


% Define the grayscale image with salt and pepper noise
noisy_image = gray;

% Define the initial window size
window_size = 3;

% Get the size of the noisy image
[rows, cols] = size(noisy_image);

% Define the maximum window size
max_window_size = 21;

% Pad the image to handle edge cases
pad_size = floor(max_window_size / 2);
padded_image = padarray(noisy_image, [pad_size, pad_size], 'replicate');

% Apply adaptive median filtering
for i = 1:rows
    for j = 1:cols
        % Get the current window centered at (i, j)
        window = padded_image(i:i+max_window_size-1, j:j+max_window_size-1);
        
        % Apply adaptive median filtering with varying window sizes
        while window_size <= max_window_size
            % Get the current window centered at (i, j) with the current window size
            current_window = padded_image(i:i+window_size-1, j:j+window_size-1);
            
            % Compute the median value of the current window
            median_value = median(current_window, 'all');
            
            % Check if the central pixel value is an impulse (salt or pepper noise)
            if noisy_image(i, j) == 0 || noisy_image(i, j) == 255
                % Check if the median value lies within the range of the current window
                if median_value > min(current_window, [], 'all') && median_value < max(current_window, [], 'all')
                    % Replace the central pixel value with the median value
                    noisy_image(i, j) = median_value;
                    break; % Move to the next pixel
                end
            else
                % Central pixel is not impulse noise, so keep its value
                break; % Move to the next pixel
            end
            
            % Increase the window size for the next iteration
            window_size = window_size + 2;
        end
        
        % Reset the window size for the next pixel
        window_size = 3;
    end
end

% Display the filtered image
imshow(noisy_image);



%Task 5 : Otsu's filter :


% Compute the histogram of the filtered image
histogram = zeros(1, 256, 'single');

for i = 1:rows
    for j = 1:cols
        intensity = noisy_image(i, j);
        histogram(intensity + 1) = histogram(intensity + 1) + 1;
    end
end
% Normalize the histogram
histogram = histogram / (rows * cols);

% Initialize variables for Otsu's method

max_variance = 0;
threshold = 0;

% Iterate over possible thresholds
for t = 1:255
    % Compute probabilities for background and foreground
    w0 = sum(histogram(1:t));
    w1 = sum(histogram(t+1:end));

    % Compute means for background and foreground
    mean0 = sum((0:t-1).*histogram(1:t)) / w0;
    mean1 = sum((t:255).*histogram(t+1:end)) / w1;
    
    % Compute between-class variance
    variance = w0 * w1 * ((mean0 - mean1)^2);
    
    % Update if variance is higher
    if variance > max_variance
        max_variance = variance;
        threshold = t - 1; % Threshold is shifted by 1 because MATLAB indexing starts from 1
    end
end
%disp(threshold);

% Binarize the image using the computed threshold
binary_image = noisy_image > threshold;

% Display the binary image
imshow(binary_image);


%Task 6 : Removing small objects :

% Define the structuring element for erosion
se = [1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1]; % Define a 3x3 structuring element (adjust as needed)

% Get the size of the binary image
[rows, cols] = size(binary_image);

% Initialize the processed binary image
binary_image_processed = false(rows, cols);

% Perform erosion
for i = 6:rows-3
    for j = 6:cols-3
        % Extract the neighborhood of the current pixel
        neighborhood = binary_image(i-3:i+3, j-3:j+3);
        
        % Check if all pixels in the neighborhood are foreground
        if all(neighborhood(:))
            % Set the current pixel to foreground in the processed image
            binary_image_processed(i, j) = true;
        end
    end
end

% Display the processed binary image
imshow(binary_image_processed);


% Define the structuring element for dilation
se = [1 1 1; 1 1 1; 1 1 1]; % Define a 3x3 structuring element (adjust as needed)

% Get the size of the binary image
[rows, cols] = size(binary_image_processed);

% Initialize the dilated binary image
dilated_binary_image = false(rows, cols);

% Perform dilation
for i = 8:rows-4
    for j = 8:cols-4
        % Extract the neighborhood of the current pixel
        neighborhood = binary_image_processed(i-4:i+4, j-4:j+4);
        
        % Check if at least one pixel in the neighborhood is foreground
        if any(neighborhood(:))
            % Set the current pixel to foreground in the dilated image
            dilated_binary_image(i, j) = true;
        end
    end
end

% Display the dilated binary image
imshow(dilated_binary_image);


%Second time dilation :

% Initialize the dilated binary image
sec_dilated_binary_image = false(rows, cols);


% Perform dilation
for i = 10:rows-5
    for j = 10:cols-5
        % Extract the neighborhood of the current pixel
        neighborhood = dilated_binary_image(i-5:i+5, j-5:j+5);
        
        % Check if at least one pixel in the neighborhood is foreground
        if any(neighborhood(:))
            % Set the current pixel to foreground in the dilated image
            sec_dilated_binary_image(i, j) = true;
        end
    end
end

% Display the dilated binary image
imshow(sec_dilated_binary_image);



%FINDING LARGEST CONNECTED COMPONENET IN IMAGE :  


function dilated_image = myDilate(binary_image)
    % Define the structuring element for dilation (3x3 square structuring element)
     se = ones(3); 
    
    % Get the size of the binary image
    [rows, cols] = size(binary_image);

    % Initialize the dilated binary image
    dilated_image = false(rows, cols);

    % Iterate over each pixel of the binary image
    for i = 1:rows
        for j = 1:cols
            % Check if the current pixel is foreground
            if binary_image(i, j)
                % Iterate over the structuring element
                for m = -3:3
                    for n = -3:3
                        % Calculate the coordinates of the neighbor pixel
                        neighbor_i = i + m;
                        neighbor_j = j + n;
                        
                        % Check if the neighbor pixel is within bounds
                        if neighbor_i >= 1 && neighbor_i <= rows && neighbor_j >= 1 && neighbor_j <= cols
                            % Set the corresponding pixel in the dilated image to foreground
                            dilated_image(neighbor_i, neighbor_j) = true;
                        end
                    end
                end
            end
        end
    end
end



% Find the first foreground pixel in the dilated binary image
[row, col] = find(dilated_binary_image, 1, 'first');

% Initialize variables
largest_component = false(size(dilated_binary_image));
visited = false(size(dilated_binary_image));

% Initialize the largest component with the first found pixel
largest_component(row, col) = true;

% Repeat until no further expansion is possible
while true
    % Perform dilation on the current largest component using custom function
    expanded_component = myDilate(largest_component);
    
    % Mark newly added foreground pixels
    new_pixels = expanded_component & ~largest_component & dilated_binary_image;
    
    % Break if no new pixels were added
    if ~any(new_pixels, 'all')
        break;
    end
    
    % Mark visited pixels
    visited = visited | new_pixels;
    
    % Update the largest component
    largest_component = visited;
end

% Display the largest connected component
imshow(largest_component);

% Calculate the area of the largest connected component
area_largest_component = sum(largest_component(:));

% Display the area
fprintf('Area of the largest connected component: %d pixels\n', area_largest_component);



