function K = cluster_det(data,min_k,max_k)
%Run k-means for a range of k
silA = zeros(1,max_k);
for k=min_k:max_k
    try
        IDX=kmeans(data,k);  %The data with two groups
        S = silhouette(data, IDX);
        silA(k)=mean(S); %The mean silhoette value for two groups
    catch
        silA(k) = 0;
    end
end
% figure
% plot(silA)
K = find(silA == max(silA(:)));
end