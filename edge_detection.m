%function img = tumor(img); %use this only when you use matlab.engine in
%python. Then make the next img as a comment
img = imread('C:\Users\HP\Downloads\archive\brain_tumor_dataset\yes\Y1.jpg');

img_re = imresize(img, [300, 300]);
img_gr_double = double(img_re);

%% thresholding

sout = imresize(img_re, [32,32]);
if size(sout,3) > 1
    sout = rgb2gray(sout);
end
t0 = 60;
th = t0 + ((max(img_re(:)) + min(img_re(:)))/2);
for i = 1:1:size(img_re,1)
    for j = 1:1:size(img_re,2)
        if img_re(i,j) > th
            sout(i,j) = 1;
        else
            sout(i,j) = 0;
        end
    end
end

%% Morphological Operation
label = bwlabel(sout);% returns label matrix that contains 8 connected components found in image
stats = regionprops(logical(sout), 'Solidity', 'Area');%returns measurements for the set of properties for each 8-connected component in the input image 
density = [stats.Solidity];
area = [stats.Area];
high_dense_area = (density>0.5);
max_area = max(area(high_dense_area));
tumor_label = find(area == max_area);
tumor1 = ismember(label, tumor_label);

% if max_area > 100
%     figure();
%     imshow(tumor1);
% end

%%
BW = edge(tumor1,'canny');
figure, imshow(BW);

%%
rgb = img_re(:,:,[1 1 1]);
red = rgb(:,:,1);
red(BW) = 255;
green = rgb(:,:,2);
green(BW) = 0;
blue = rgb(:,:,3);
blue(BW) = 0;

tumorOutlineInserted(:,:,1) = red;
tumorOutlineInserted(:,:,2) = green;
tumorOutlineInserted(:,:,3) = blue;

figure();
imshow(tumorOutlineInserted);
title('Detected Tumor');
