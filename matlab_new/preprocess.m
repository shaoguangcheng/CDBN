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
    ratio = min([512/size(image,1), 512/size(image,2), 1]);
    if ratio<1
        image = imresize(image, [round(ratio*size(image,1)), round(ratio*size(image,2))], 'bicubic');
    end
    
    images(:,:,i) = image;
end

end


function images = whiten(images)
N = size(images, 3);

for i = 1 : N
    im = images(:,:,i);
    if size(im,3)>1, im = rgb2gray(im); end
    im = double(im);

    im = im - mean(im(:));
    im = im./std(im(:));

    N1 = size(im, 1);
    N2 = size(im, 2);

    [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
    rho=sqrt(fx.*fx+fy.*fy)';

    f_0=0.4*mean([N1,N2]);
    filt=rho.*exp(-(rho/f_0).^4);

    If=fft2(im);
    imw=real(ifft2(If.*fftshift(filt)));

    im_out = imw/std(imw(:)); % 0.1 is the same factor as in make-your-own-images
    
    im_out = im_out - mean(mean(im_out));
    im_out = im_out/sqrt(mean(mean(im_out.^2)));
    im_out = sqrt(0.1)*im_out;
    images(:,:,i) = im_out;
end

end
