function [ R_train,z_train, R_test, z_test,a,b] = select_sample( R_train,z_train, R_test, z_test, knowledge, select_min,randomise)
%Select Sample 
%   Select sample from criteria


    if select_min == 0
        knowledge = knowledge - min(knowledge(:));
        knowledge = knowledge/max(knowledge(:));
        knowledge = 1 - knowledge;
    end
    
    % Get all positive indices in R_test
    [i_x,i_y] = find(z_test == 1);
    knowledge = knowledge.*z_test+max(knowledge(:))*(1-z_test);


    
%     [a,b] = find(knowledge==min(min(knowledge(i_x,i_y))));
    [a,b] = find(knowledge==min(min(knowledge)));
    if randomise
        j = randi(length(a));
    else
        j=1 ;
    end
    a=a(j);
    b=b(j);
    while ((z_train(a,b)==1)||(z_test(a,b)==0));
        knowledge(a,b) = max(knowledge(:));
%         [a,b] = find(knowledge==min(min(knowledge(i_x,i_y))));
        [a,b] = find(knowledge==min(min(knowledge)));
        if randomise
            j = randi(length(a));
        else
            j=1 ;
        end
        a=a(j);
        b=b(j);
    end


%Swap
R_train(a,b) = R_test(a,b);
z_train(a,b) = 1;
R_test(a,b) = 0;
z_test(a,b) = 0;
end

