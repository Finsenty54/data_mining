function p=chi_fang(image)
%计算各像素值出现次数
Cdata=reshape(image,1,[]);
tbl=tabulate(Cdata);
values=tbl(:,1);
my_count=tbl(:,2);

%r为卡方统计量，k-1为自由度
r=0;k=0;
for i=0:127
    tmp=0.5*(my_count(values==2*i)+my_count(values==(2*i+1)));
    if tmp~=0
        r=r+((my_count(values==2*i)-tmp)^2)/tmp;
        k=k+1;
    end
end
%Chi-square probability density function
p=1-chi2pdf(r,k-1);
%p 接近于 1，说明有秘密信息