%____________________________________________
% Clustered Knowledge Search                 |
%____________________________________________|
% Seb Grubb - sg3510@ic.ac.uk                |
%____________________________________________|
% This function returns a knowledge matrix of|
% areas with a rating of their knowledge     |
% on the clusters the the feature matrices.  |
%====================Inputs==================|
% - U,V: feature matrices learnt from R      |
% - z_m: mask matrix, 1 for known value, 0   |
%        otherwise                           |
%====================Outputs=================|
% - knowledge: matrix of same size as R with |
%              values representing the amount|
%              known of the cell based on    |
%              cluster.                      |
%____________________________________________|
function [ knowledge ] = CKS_UV_cluster(U,V,z_m)
% Cluster UV by features
% Determine best cluster size
k_u = cluster_det(U,2,round(2*log(length(U))));
k_v = cluster_det(V,2,round(2*log(length(V))));
% Error handling
while 1
    try
        u_clusters = kmeans(U,k_u);
        v_clusters = kmeans(V,k_v);
        break
    catch
        k_u = k_u - 1;
        k_v = k_v - 1;
        fprintf('Reducing cluster size k_u = %d k_v = %d',k_u,k_v)
    end
end
% create knowledge variables
u_info = mean(z_m,2); u_knowledge = zeros(1,k_u);
v_info = mean(z_m,1); v_knowledge = zeros(1,k_u);
for i =1:k_u
    u_knowledge(i) = sum(u_info(u_clusters==i));
end

for i =1:k_v
    v_knowledge(i) = sum(v_info(v_clusters==i));
end

%create knowledge matrix
a = zeros(length(U),1);
b = zeros(length(V),1);

% Assign amount known
for i =1:k_u
    a(u_clusters==i) = u_knowledge(i);
end
for i =1:k_v
    b(v_clusters==i) = v_knowledge(i);
end

knowledge = a*b';

end

