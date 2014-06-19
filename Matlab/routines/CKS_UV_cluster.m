function [ knowledge ] = CKS_UV_cluster(U,V,z_m)
%% Cluster UV by features
k_u = cluster_det(U,2,round(2*log(length(U))));
k_v = cluster_det(V,2,round(2*log(length(V))));
%cluster use and movies then select users and movies we know the least
%about
% k_u = 5;
% k_v = 5;
while 1
    try
        u_clusters = kmeans(U,k_u);
        v_clusters = kmeans(V,k_v);
        break
    catch
        k_u = k_u - 1;
        k_v = k_v - 1;
        fprintf('k_u = %d k_v = %d',k_u,k_v)
    end
end
% u_clusters = kmeans(U,k_u);
% v_clusters = kmeans(V,k_v);


u_info = mean(z_m,2);
u_knowledge = zeros(1,k_u);
v_info = mean(z_m,1);
v_knowledge = zeros(1,k_u);
for i =1:k_u
    u_knowledge(i) = sum(u_info(u_clusters==i));%/length(u_info(u_clusters==i));
end
% u_knowledge = u_knowledge/max(u_knowledge);
for i =1:k_v
    v_knowledge(i) = sum(v_info(v_clusters==i));%/length(v_info(v_clusters==i));
end
% v_knowledge = v_knowledge/max(v_knowledge);
%create knowledge matrix
a = zeros(length(U),1);
b = zeros(length(V),1);
for i =1:k_u
    a(u_clusters==i) = u_knowledge(i);%/sum(u_knowledge(:));
end
for i =1:k_v
    b(v_clusters==i) = v_knowledge(i);%/sum(v_knowledge(:));
end
knowledge = a*b';
% knowledge = knowledge - min(min(knowledge));
% knowledge=knowledge/max(max(knowledge));

end

