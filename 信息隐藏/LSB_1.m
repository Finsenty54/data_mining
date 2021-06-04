%https://www.zhihu.com/question/301957087
%https://wenku.baidu.com/view/36ee7282f01dc281e43af00e.html
%https://hejueyun.github.io/posts/42a865c0/#more
%https://blog.csdn.net/HizT_1999/article/details/106951364

clc;clear;close all;

cover_image=imread('peppers.tif');
cover_image=rgb2gray(cover_image);
% figure;
% subplot(121),imagesc(cover_image); colormap gray;
% axis off; axis square;

[row,col]=size(cover_image);
message=randi([0,1],row,col); 
%LSB«∂»Î
insert_image=bitset(cover_image,1,message);

% subplot(122),imagesc(insert_image); colormap gray;
% axis off; axis square;

p_c=chi_fang(cover_image);
p_i=chi_fang(insert_image);