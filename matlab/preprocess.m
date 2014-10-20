function [ images ] = preprocess(dataPath)
%%
images = readImages(dataPath);
images = whiten(images);

end

function [images] = readImages(dataPath)
%%
imageNames = importdata(dataPath);
N = size(imageNames,1);

for i = 1 : N
    image = imread(imageNames{i});
    if size(image,3)>1
        image = double(rgb2gray(image));
    else
        image = double(image);
    end
    
    %resize images
    ratio = min([70/size(image,1), 70/size(image,2), 1]);
    if ratio<1
        image = imresize(image, [round(ratio*size(image,1)), round(ratio*size(image,2))], 'bicubic');
    end
    
    images(:,:,i) = image;
end

end

function [images] = whiten(images)
%%
N = size(images, 3);
for i = 1:N
    image = images(:,:,i);
    image = image - mean(image(:));
    image = image./std(image(:));

%     N1 = size(image, 1);
%     N2 = size(image, 2);
% 
%     [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
%     rho=sqrt(fx.*fx+fy.*fy)';
% 
%     f_0=0.4*mean([N1,N2]);
%     filt=rho.*exp(-(rho/f_0).^4);
% 
%     If=fft2(image);
%     imagew=real(ifft2(If.*fftshift(filt)));
% 
%     imagew = imagew/std(imagew(:));
    


    images(:,:,i) = image;
    imshow(image);
end
end