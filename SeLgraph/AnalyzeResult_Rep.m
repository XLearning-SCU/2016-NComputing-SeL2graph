clc;
fprintf(['|For the data set ' CurData '\n']);

if isfield(options,'lambda')
    Rec = SRC_rec;
    time = SRC_time;
    fprintf(' * The mean accuracy  of SRC is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of SRC is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf(' | when lambda = %f, ReducedDim = %d\n' ,options.lambda(1), options.k(1))

    Rec = SVM_rec;
    time = SVM_time;
    fprintf(' * The mean accuracy  of SVM is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of SVM is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf(' | when lambda = %f, ReducedDim = %d\n', options.lambda(3), options.k(3));

    Rec = KNN_rec;
    time = KNN_time;
    fprintf(' * The mean accuracy  of NN is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of NN is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf(' | when lambda = %f, ReducedDim = %d\n', options.lambda(4), options.k(4));

else
    Rec = SRC_rec;
    time = SRC_time;
    fprintf(' * The mean accuracy of  SRC is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of SRC is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf([' | when lambda = ' num2str(options.k(1)) '\n'])


    Rec = SVM_rec;
    time = SVM_time;
    fprintf(' * The mean accuracy of  SVM is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of SVM is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf([' | when lambda = ' num2str(options.k(3)) '\n']);

    Rec = KNN_rec;
    time = KNN_time;
    fprintf(' * The mean accuracy of  NN is about %2.2f and the std is about %2.2f\n', mean(Rec*100), std(Rec*100));
    fprintf(' | The mean time cost of NN is about %2.2f and the std is about %2.2f\n', mean(time), std(time));
%     fprintf([' | when lambda = ' num2str(options.k(4)) '\n'])
end

